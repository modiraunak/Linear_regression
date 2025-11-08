import numpy as np 
import random 

num_states = 16
num_actions = 4
q_table = np.zeros((num_states, num_actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.2
num_episodes = 1000

#define a simple reward structure
rewards = np.zeros(num_states)
rewards[15] = 1  # Goal state

#function to determine the next state based on current state and action
def get_next_state(state, action):
    if action == 0:  # up
        return max(0, state - 4)
    elif action == 1:  # down
        return min(num_states - 1, state + 4)
    elif action == 2:  # left
        return state - 1 if state % 4 != 0 else state
    elif action == 3:  # right
        return state + 1 if state % 4 != 3 else state
    return state
for episode in range(num_episodes):
    state = random.randint(0, num_states - 1)
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            action = np.argmax(q_table[state])
        next_state = get_next_state(state, action)
        reward = rewards[next_state]
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state
        if state == 15:
            done = True
print("Trained Q-Table:")
print(q_table)