
# Import dependencies
# pip install stable-baselines3[extra]

import os
import gym # open AI gym
from stable_baselines3 import PPO # PPO is the model
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# 2. Load environment
environment_name = "CartPole-v0" # environment name to pre-installed Open AI gym
env = gym.make(environment_name)

episodes = 5 # each episode is one full game within the environment
for episode in range(1, episodes+1): # looping thru each episode
    state = env.reset() # get initial set of observations (agent, action, observation, reward)
    done = False # temp variables
    score = 0 # running score counter

    while not done:
        env.render() # render allows us to view the graphical representation
        action = env.action_space.sample() # generate a random action
        n_state, reward, done, info = env.step(action)
        score+=reward # accumulate reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

# Understanding the environment
# 0-push cart to left, 1-push cart to the right
env.action_space.sample()

# [cart position, cart velocity, pole angle, pole angular velocity]
env.observation_space.sample()

# 3. Train an RL Model
log_path = os.path.join('Training','Logs')
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log = log_path) # create algorithm / agent, pass thru parameters
# Train the model
model.learn(total_timesteps=20000)

# 4. Save and Reload Model
print('Save and Reload Model')
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model') # path variable
model.save(PPO_path) # save Model
del model
# print(PPO_path)
model = PPO.load(PPO_path, env=env) # load the model back up

# 4. Testing and evaluation
from stable_baselines3.common.evaluation import evaluate_policy
evaluate_policy(model, env, n_eval_episodes=10, render=True)

env.close()

# episode mean length -> on average how long a particular episode lasted
# episode reward mean -> the aveage reward that the agent accumulated per episode


# 5. Test Models

obs = env.reset()
while True:
    # agent = model
    action, _states = model.predict(obs) # use model.predict, Now using the model
    obs, rewards, done, info = env.step(action) # 4 values (cart position, Cart velocity, Pole angle, pole angular velocity)
    env.render() # environment
    if done:
        print('info', info)
        break

env.close()

# 6. Viewing Logs in tensorboard_log
#training_log_path = os.path.join(log_path, 'PPO_3')
#!tensorboard --logdir={training_log_path}

# 7. Adding a callback to the training Stage
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os

save_path = os.path.join('Training', 'Saved Models')
log_path = os.path.join('Training', 'Logs')

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])


stop_callback = StopTrainingOnRewardThreshold(reward_threshold=190, verbose=1)

# save the best new model every 10k time steps
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=save_path,
                             verbose=1)

# create new PPO model and assign call callbacks
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)
# but this time round, pass through the callback
model.learn(total_timesteps=20000, callback=eval_callback)

model_path = os.path.join('Training', 'Saved Models', 'best_model')
model = PPO.load(model_path, env=env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

# Changing policies
net_arch=[dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])]

model = PPO('MlpPolicy', env, verbose = 1, policy_kwargs={'net_arch': net_arch})
model.learn(total_timesteps=20000, callback=eval_callback)

#9 Using an alternate algorithm
# Use DQN instead of PPO

# import DQN
from stable_baselines3 import DQN

model = DQN('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)

model.learn(total_timesteps=20000, callback=eval_callback)
dqn_path = os.path.join('Training', 'Saved Models', 'DQN_model')
model.save(dqn_path)
model = DQN.load(dqn_path, env=env)
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()
