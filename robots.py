import numpy as np
import gym
from gym.envs.registration import register, spec
import time 


class gym_mountaincar(object):

    def __init__(self, render=False, temp=False):

        self.render = render
        self.temp = temp
        self.MY_ENV_NAME = 'MountainCarContinuous-v0'
        self.env = gym.make(self.MY_ENV_NAME)
        #
        self.action_dim = 1#self.env.action_space
        self.state_dim = 2#self.env.observation_space
        self.state = self.env.reset()
        self.reward = -1.
        self.info = []
        self.action = []
        self.done = False
        self.step = 0
        self.goal = 0


    def update(self, action):
        
        self.action = [action]
        self.state, self.reward, self.done, self.info = self.env.step(self.action)
        self.step = self.step + 1
        #if self.state == (some goal measurement):
        #    # I reached the goal
        #    self.goal = self.goal + 1

        self.development() # this is just in case you want to render, or slow down the execution
        return self.state, self.reward, self.done, self.step

    def reset(self):
        self.state = self.env.reset()
        self.reward = -1.
        self.info = []
        self.action = []
        self.done = False
        self.step = 0
        return self.state, self.done, self.step 

    def development(self):
        if self.render: 
            self.env.render()
        if self.temp:
            if (self.step %10 == 0):
                time.sleep(1) 