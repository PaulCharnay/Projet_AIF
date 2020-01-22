import os
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import A2C


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
    """Trains an A2C policy """
    env = create_env()

    model = A2C(
        policy = CnnPolicy,
        env = env,
        gamma = 0.99,
        n_steps = 5,
        vf_coef=0.25, 
        ent_coef=0.01,
        max_grad_norm=0.5,
        learning_rate=7e-4,
        alpha=0.99,
        epsilon=1e-05,
        lr_schedule='constant',
        verbose=1,
        tensorboard_log="./tb"  
    )

    model.learn(
        total_timesteps=int(1e7), 
        callback=callback, 
        tb_log_name="a2c"
    )

    model.save("models/pacman_a2c.pkl")



if __name__ == '__main__':
    os.makedirs("/models", exist_ok=True)
    os.makedirs("/tb", exist_ok=True)
    n_steps = 1
    train()