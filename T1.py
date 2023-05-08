class ValueIterationAgent:
    def __init__(self, states, transition_probs, rewards, gamma=0.9, theta=0.01):
        self.states = states
        self.P = transition_probs
        self.R = rewards
        self.gamma = gamma
        self.theta = theta
        self.V = {s: 0 for s in self.states}
        
    def learn(self):
        while True:
            delta = 0
            for s in self.states:
                v_old = self.V[s]
                self.V[s] = 0
                for s_next in self.states:
                    self.V[s] += (self.P[s][s_next] * (self.R[s_next] + self.gamma * v_old))
                delta = max(delta, abs(v_old - self.V[s]))
            print(self.V)
            if delta < self.theta:
                break

states = ['s1', 's2', 's3', 's4', 's5', 's6']

# 构建马尔可夫决策过程
transition_probs = {
    's1': {'s1': 0.1, 's2': 0, 's3': 0, 's4': 0.4, 's5': 0.5, 's6': 0},
    's2': {'s1': 0.1, 's2': 0.2, 's3': 0.2, 's4': 0, 's5': 0.5, 's6': 0},
    's3': {'s1': 0, 's2': 0.1, 's3': 0.3, 's4': 0, 's5': 0, 's6': 0.6},
    's4': {'s1': 0.1, 's2': 0, 's3': 0, 's4': 0.9, 's5': 0, 's6': 0},
    's5': {'s1': 0, 's2': 0, 's3': 0, 's4': 0.4, 's5': 0, 's6': 0.6},
    's6': {'s1': 0, 's2': 0, 's3': 0, 's4': 0, 's5': 0.5, 's6': 0.5},
}
rewards = {
    's1': 3,
    's2': 1,
    's3': 2,
    's4': 10,
    's5': 4,
    's6': 3
}
gamma = 0.9

# 构建ValueIterationAgent
agent = ValueIterationAgent(states,transition_probs, rewards, gamma)

# 进行价值迭代
agent.learn()


# 打印状态价值函数
print("State value function:")
for state, value in agent.V.items():
    print(f"{state}: {value}")