import numpy as np
import pandas as pd
from task import Task

class QLearningAgent():
    def __init__(self, task, learning_rate=0.01, gamma=0.9, e_greedy=0.9):
        # Task (environment) information
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = e_greedy
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        self.q_table = dict()

        self.w = np.random.normal(
            size=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
            scale=(self.action_range / (2 * self.state_size))) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()
    
    def act(self, state):
        # Choose action based on given state and policy
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table[state]
            action = state_action
        else:
            action = np.random.choice()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def learn(self, state, action, reward, next_state):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = reward + self.gamma * self.  # equal noise in all directions
        