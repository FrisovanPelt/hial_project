import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
import os
import sys
from collections import deque
from pref_learn import prepare_demo_pool
from pref_learn import feature_function

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR + '/envs/')
sys.path.append(PARENT_DIR + '/utils/')

from task_envs import PnPNewRobotEnv
from env_wrappers import ActionNormalizer, ResetWrapper, TimeLimitWrapper, reconstruct_state

# Load learned feature weights
WEIGHT_FILE = "final_feature_weights.csv"
def load_weights():
    with open(WEIGHT_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        return np.array([float(row[0]) for row in reader])

# Define Deep Q-Network
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define DQfD Agent
class DQfDAgent:
    def __init__(self, state_dim, action_dim, replay_buffer, learning_rate=1e-4, gamma=0.99):
        self.q_network = DQNetwork(state_dim, action_dim)
        self.target_network = DQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.9995  # Decay rate
        self.epsilon_min = 0.1  # Minimum exploration

    def select_action(self, state):
        if random.random() < self.epsilon:
            return np.random.uniform(-1, 1, size=(4,))  # Continuous action space
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy().flatten()
    
    def update_policy(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        q_values = self.q_network(states)
        target_q_values = self.target_network(next_states)
        max_next_q = target_q_values.max(dim=1, keepdim=True)[0]
        expected_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        loss = torch.nn.functional.mse_loss(q_values, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Prepare environment
env = PnPNewRobotEnv(render=True)
env = ActionNormalizer(env)
env = ResetWrapper(env=env)
env = TimeLimitWrapper(env=env, max_steps=150)

# Load expert demonstrations
demo_path = PARENT_DIR + '/demo_data/PickAndPlace/'
demos = prepare_demo_pool(demo_path)

# Initialize Replay Buffer with expert demonstrations
replay_buffer = deque(maxlen=100000)
for demo in demos:
    for i in range(len(demo['state_trajectory']) - 1):
        replay_buffer.append((
            reconstruct_state(demo['state_trajectory']),
            demo['action_trajectory'][i],
            demo['reward_trajectory'][i],
            reconstruct_state(demo['next_state_trajectory']),
            demo['done_trajectory'][i]
        ))

# Initialize agent
state_dim = len(reconstruct_state(demos[0]['state_trajectory'][0]))
action_dim = len(demos[0]['action_trajectory'][0])
agent = DQfDAgent(state_dim, action_dim, replay_buffer)

# Train policy
num_steps = 500000
save_interval = 1000
batch_size = 64

success_rates = []
for step in range(num_steps):
    state = reconstruct_state(env.reset())
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = reconstruct_state(next_state)
        reward = np.dot(load_weights(), feature_function([(state, action)]))  # Use learned reward
        replay_buffer.append((state, action, reward, next_state, done))
        agent.update_policy(batch_size)
        state = next_state
    
    if step % save_interval == 0:
        torch.save(agent.q_network.state_dict(), f"policy_model_{step}.pth")
        
        # Evaluate success rate
        successes = []
        for _ in range(10):
            test_state = reconstruct_state(env.reset())
            test_done = False
            while not test_done:
                test_action = agent.select_action(test_state)
                test_state, _, test_done, test_info = env.step(test_action)
                test_state = reconstruct_state(test_state)
            successes.append(test_info['is_success'])
        success_rate = np.mean(successes)
        success_rates.append((step, success_rate))
        print(f"Step {step}: Success Rate: {success_rate}")

# Save learning curve
import matplotlib.pyplot as plt
steps, rates = zip(*success_rates)
plt.plot(steps, rates)
plt.xlabel("Training Steps")
plt.ylabel("Success Rate")
plt.title("Policy Learning Curve")
plt.savefig("learning_curve.png")
plt.show()