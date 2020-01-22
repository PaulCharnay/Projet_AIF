import os
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines import ACER


def create_env():
    """Creation of MsPacman environment"""
    #ACER needs the same number of environments as for training. 
    env = make_atari_env("MsPacmanNoFrameskip-v0", num_env=16, seed=817) 
    env = VecFrameStack(env, n_stack=4)
    return env

def wrap_video_env(env, name, video_length, path = 'videos/'):
    """Wrap the environment to record a video"""
    env = VecVideoRecorder(env, path,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=name)
    return env

def record_video():
    """Record of a video for an trained ACER agent"""
    model = ACER.load("models/pacman_acer.pkl", verbose=1)
    env = create_env()
    model.set_env(env)
    
    video_length = 3000
    env = wrap_video_env(env, name = "pacman_acer", video_length = video_length, path = 'videos/')

    state = env.reset()
    for _ in range(video_length + 1):
        action, _states = model.predict(state)
        state, _, _, _ = env.step(action)
    print("Video recorded")
    env.close()



if __name__ == '__main__':
    os.makedirs("/videos", exist_ok=True)
    record_video()