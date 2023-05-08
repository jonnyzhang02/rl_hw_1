'''
Author: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
Date: 2023-04-21 08:41:23
LastEditors: jonnyzhang02 71881972+jonnyzhang02@users.noreply.github.com
LastEditTime: 2023-05-08 17:20:21
FilePath: \rl_hw_1\T2.py
Description: coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn

Copyright (c) 2023 by zhangyang0207@bupt.edu.cn, All Rights Reserved. 
'''
import numpy as np
import time


class ValueIterationAgent:
    def __init__(self, states, actions, transition_probs, rewards, discount_factor, threshold):
        self.states = states
        self.actions = actions
        self.P = transition_probs
        self.R = rewards
        self.gamma = discount_factor 
        self.theta = threshold
        self.V = {s: 0 for s in self.states}

    def value_iteration(self):
        while True:
            delta = 0
            for state in self.states:
                v = self.V[state]
                self.V[state] = max([sum([self.P[state][action][next_state] * (self.R[state][action][next_state] + self.gamma * self.V[next_state]) for next_state in self.states]) for action in self.actions])
                delta = max(delta, abs(v - self.V[state]))
            if delta < self.theta:
                break
        policy = {}
        for state in self.states:
            policy[state] = max(self.actions, key=lambda action: sum([self.P[state][action][next_state] * (self.R[state][action][next_state] + self.gamma * self.V[next_state]) for next_state in self.states]))
        return policy


class PolicyIterationAgent:
    def __init__(self, states, actions, transition_probs, rewards, discount_factor, threshold):
        self.states = states
        self.actions = actions
        self.P = transition_probs
        self.R = rewards
        self.gamma = discount_factor
        self.theta = threshold
        self.policy = {s: self.actions[np.random.randint(len(self.actions))] for s in self.states}
        self.V = {s: 0 for s in self.states}

    def policy_evaluation(self, policy):
        while True:
            delta = 0
            for state in self.states:
                v = self.V[state]
                self.V[state] = sum([self.P[state][policy[state]][next_state] * (self.R[state][policy[state]][next_state] + self.gamma * self.V[next_state]) for next_state in self.states])
                delta = max(delta, abs(v - self.V[state]))
            if delta < self.theta:
                break
        return self.V

    def policy_improvement(self, values):
        policy_stable = True
        for state in self.states:
            old_action = self.policy[state]
            self.policy[state] = max(self.actions, key=lambda action: sum([self.P[state][action][next_state] * (self.R[state][action][next_state] + self.gamma * values[next_state]) for next_state in self.states]))
            if old_action != self.policy[state]:
                policy_stable = False
        return policy_stable

    def policy_iteration(self):
        while True:
            self.policy_evaluation(self.policy)
            policy_stable = self.policy_improvement(self.V)
            if policy_stable:
                return self.policy
            
# 定义状态、行动、状态转移概率和奖励
states = ['s1', 's2', 's3']
actions = ['a', 'b']
transition_probs = {
    's1': {
        'a': {'s1': 0.5, 's2': 0.5, 's3': 0},
        'b': {'s1': 0.1, 's2': 0.9, 's3': 0}
    },
    's2': {
        'a': {'s1': 0.7, 's2': 0.3, 's3': 0},
        'b': {'s1': 0.5, 's2': 0.5, 's3': 0}
    },
    's3': {
        'a': {'s1': 1.0, 's2': 0.0, 's3': 0},
        'b': {'s3': 1.0, 's2': 0.0, 's1': 0}
    }
}
rewards = {
    's1': {
        'a': {'s1': 0, 's2': 10, 's3': 0},
        'b': {'s1': 0, 's2': 5, 's3': 0}
    },
    's2': {
        'a': {'s1': 0, 's2': 2, 's3': 0},
        'b': {'s1': 0, 's2': 1, 's3': 0}
    },
    's3': {
        'a': {'s1': 0, 's2': 0, 's3': 0},
        'b': {'s3': 100, 's2': 0, 's1': 0}
    }
}
discount_factor = 0.95
threshold = 0.0001

agent = ValueIterationAgent(states, actions, transition_probs, rewards, discount_factor, threshold)
policy = agent.value_iteration()

print("Optimal policy:")
print(policy)


