# pip3 install torch torchvision torchaudio



import gym #Open AI Gym
from stable_baselines3 import A2C #different algorithm
from stable_baselines3.common.vec_env import VecFrameStack # train on 4 environemnts at the same time
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env # work with atari environments
import os

#python -m atari_py.import_roms .\ROMS\ROMS
-m atari_py.import_roms .\ROMS\ROMS

environment_name = "Breakout-v0"
env = gym.make(environment_name)

episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

env.action_space.sample()
env.observation_space.sample()
