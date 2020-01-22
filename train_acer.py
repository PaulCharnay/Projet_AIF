import os
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import ACER


def create_env():
    """Creation of MsPacman environment"""
    env = make_atari_env("MsPacmanNoFrameskip-v0", num_env=16, seed=817)
    env = VecFrameStack(env, n_stack=4)
    return env 

def callback(_locals, _globals):
    """Save the model every 100 calls """
    global n_steps
    if (n_steps) % 100 == 0:
        print("Saving model after {} steps".format(n_steps))
        _locals['self'].save("./models/tmp_model.pkl")
    n_steps += 1
    return True

def train():
    """Trains an ACER policy """
    env = create_env()

    model = ACER(
        policy = CnnPolicy,
        env = env,
        gamma=0.99,
        n_steps=20,
        num_procs=4,
        q_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=10,
        learning_rate=0.0007,
        lr_schedule='linear',
        rprop_alpha=0.99,
        rprop_epsilon=1e-05,
        buffer_size=5000,
        replay_ratio=4,
        replay_start=1000,
        correction_term=10.0,
        trust_region=True,
        alpha=0.99,
        delta=1,
        verbose=1,
        tensorboard_log="./tb"
    )

    model.learn(
        total_timesteps=int(1e7), 
        callback=callback, 
        tb_log_name="acer"
    )

    model.save("models/pacman_acer.pkl")



if __name__ == '__main__':
    os.makedirs("/models", exist_ok=True)
    os.makedirs("/tb", exist_ok=True)
    n_steps = 1
    train()