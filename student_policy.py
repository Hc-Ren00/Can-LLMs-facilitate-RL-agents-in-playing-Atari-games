import gymnasium as gym
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sampleEnv import MazeGameEnv
import torch

from gym_minigrid.wrappers import *
import gym as gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.callbacks import BaseCallback


class State_value_Callback(BaseCallback):
  def __init__(self, verbose: int = 1):
    super(State_value_Callback, self).__init__(verbose=verbose)
    self.value_function = {}

  def _on_step(self) -> bool: #will be called by the model after each call to 'env.step()'.
    # print(self.model._last_obs)
    curr_State=self.model._last_obs
    if curr_State.ndim==2:
      curr_State=curr_State[0]
    a,b = curr_State
    state = (int(a),int(b))#self.training_env.get_attr('envs')[0].state  #use this to get state information
    if type(state) != "tuple":
       state = tuple(state)
    value = (self.model.policy.predict_values(torch.Tensor(self.model._last_obs))).tolist()[0][0]   #get the state value
    if state in self.value_function.keys():
      self.value_function[state] = (value, self.value_function[state][1] + 1)
    else:
      self.value_function[state] = (value, 1)
    return True
  
class PPOAlgo:
  def __init__(self, total_timesteps, env, model=None):
    self.model = model
    self.tt = total_timesteps
    self.callback = State_value_Callback()
    self.env = env

  def init_model(self):
    if self.model is None:
      self.model = PPO("MlpPolicy", self.env, verbose=1)

  def train(self,total_timesteps):
    self.model.learn(total_timesteps=total_timesteps, callback=self.callback)

  def test(self):
    print(self.callback.value_function)
    obs = self.env.reset()
    rewards = 0
    done = [False]  #change it if n_env != 1
    n=0
    while not any(done) == True:
        n+=1
        action, _states = self.model.predict(obs)
        obs, reward, done, info = self.env.step(action)
        rewards += reward
        self.env.render()
    print(rewards,n)


# maze = [
#     ['S', '.', '.'],
#     ['.', '.', '.'],
#     ['W', '.', 'G'],
# ]

# #env = MazeGameEnv(maze=maze)
# vec_env = make_vec_env(MazeGameEnv, n_envs=1, env_kwargs={'maze':maze})
# Callback = State_value_Callback()

# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=1000, callback=Callback)
# print(Callback.value_function)
# obs = vec_env.reset()
# rewards = 0
# done = [False]  #change it if n_env != 1
# n=0
# while not any(done) == True:
#     n+=1
#     action, _states = model.predict(obs)
#     obs, reward, done, info = vec_env.step(action)
#     rewards += reward
#     vec_env.render()
# print(rewards,n)

# #can get all parameters for the mlp
# # print(model.get_parameters())

