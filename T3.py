'''
Author: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
Date: 2023-05-08 17:25:04
LastEditors: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
LastEditTime: 2023-05-08 20:30:50
FilePath: \rl_hw_1\T3.py
Description: coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn

Copyright (c) 2023 by zhangyang0207@bupt.edu.cn, All Rights Reserved. 
'''
import numpy as np

class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha, gamma, epsilon):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # exploration rate
        self.reward_history = []

    def choose_action(self, state, is_random=True):
        if is_random and np.random.uniform(0, 1) < self.epsilon:
            # Random action
            action = np.random.choice(self.Q.shape[1])
        else:
            # Greedy action
            action = np.argmax(self.Q[state, :])
        return action

    def update(self, state, action, next_state, reward):
        # Q-Learning update rule
        self.Q[state, action] = self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action])

    def train(self, env, num_episodes):
        for i in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Choose action
                action = self.choose_action(state)

                # Take action
                next_state, reward, done, _ = env.step(action)

                # Update Q-value
                self.update(state, action, next_state, reward)

                # Update state and episode reward
                state = next_state
                episode_reward += reward

            # Update epsilon
            self.epsilon *= 0.99

            # Record episode reward
            self.reward_history.append(episode_reward)

            # Print progress
            if (i+1) % 100 == 0:
                print("Episode {}/{}. Last 100 episodes average reward: {:.2f}".format(i+1, num_episodes, np.mean(self.reward_history[-100:])))

    def test(self, env):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Choose action
            action = self.choose_action(state, is_random=False)

            # Take action
            next_state, reward, done, _ = env.step(action)

            # Update state and episode reward
            state = next_state
            episode_reward += reward

        return episode_reward
