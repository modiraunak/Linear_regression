import gymnasium as gym 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

#Define the DQN model 
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, output_size)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)
    
#Hyperparameters
env = gym.make('CartPole-v1')
learning_rate = 0.001
gamma = 0.99
buffer_size = 10000
batch_size = 32
eplision = 0.1
target_update_freq = 100

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(input_size, output_size).to(device)
target_net = DQN(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

replay_buffer = []
def train(num_episodes):
    step_count = 0

    for episode in range(num_episodes):
        state, _= env.reset()
        state = np.array(state)
        
        done = False
        total_reward = 0
        while not done:
            step_count += 1
            if random.random() < eplision:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state)
            total_reward += reward
            
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > buffer_size:
                replay_buffer.pop(0)
            
            state = next_state
            
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states_tensor = torch.FloatTensor(states).to(device)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states_tensor = torch.FloatTensor(next_states).to(device)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                current_q_values = policy_net(states_tensor).gather(1, actions_tensor)
                with torch.no_grad():
                    max_next_q_values = target_net(next_states_tensor).max(1)[0].unsqueeze(1)
                    target_q_values = rewards_tensor + (gamma * max_next_q_values * (1 - dones_tensor))
                
                loss = criterion(current_q_values, target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if step_count % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode+1}, Total Reward: {total_reward}")
train(500)
env.close()

