import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym

# Setup environment
env = gym.make('CartPole-v1')
# Global parameters
learning_rate = 0.001
gamma = 0.99
state_shape = env.observation_space.shape[0]
num_actions = env.action_space.n

# Policy network
def build_policy_model():
    # Use the Keras Functional API for clearer definition, or Sequential as used by user.
    # Sticking to Sequential as per user request.
    model = tf.keras.Sequential([
        layers.Input(shape=(state_shape,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(num_actions, activation='softmax')
    ])
    # The loss here is technically unused as we compute the loss with custom gradients,
    # but the optimizer is needed.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model
policy_model = build_policy_model()

# Function to choose action based on policy
def choose_action(state):
    # state is already a NumPy array when passed from the main loop
    state = state.reshape([1, state_shape]).astype(np.float32)
    # Get probabilities from the model
    probabilities = policy_model(state)
    # Sample an action from the probability distribution
    return np.random.choice(num_actions, p=probabilities[0].numpy())

# Function to calculate the discounted returns (G_t) and normalize them (standardization)
def discounted_rewards(rewards):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + gamma * cumulative
        discounted[i] = cumulative
    # Subtract mean and divide by standard deviation for normalization (baseline subtraction)
    # Using np.finfo(np.float32).eps for stability
    std_dev = np.std(discounted)
    if std_dev == 0:
        return discounted
    return (discounted - np.mean(discounted)) / std_dev

# Training function
def train_on_episodes(states, actions, rewards):
    # 1. Calculate the normalized discounted rewards (Advantages)
    disc_rewards = discounted_rewards(rewards)

    with tf.GradientTape() as tape:
        # 2. Get the action probabilities for all states
        action_probs = policy_model(states, training=True)
        
        # 3. Create indices to gather the probability of the taken action for each step
        # actions must be converted to int32 for tf.stack/tf.range
        actions = tf.cast(actions, tf.int32)
        action_indices = tf.stack([tf.range(tf.shape(actions)[0], dtype=tf.int32), actions], axis=1)
        
        # 4. Gather the probabilities of the actions actually taken
        selected_action_probs = tf.gather_nd(action_probs, action_indices)
        
        # 5. Calculate the REINFORCE loss: - log(pi(a|s)) * G_t
        # This is where the original code had an error: it used the function name instead of the variable.
        loss = -tf.reduce_mean(tf.math.log(selected_action_probs) * disc_rewards)
    
    # 6. Apply gradients
    gradients = tape.gradient(loss, policy_model.trainable_variables)
    policy_model.optimizer.apply_gradients(zip(gradients, policy_model.trainable_variables))

# Main training loop
num_episodes = 1000
for episode in range(num_episodes):
    # env.reset() returns (observation, info)
    state, _ = env.reset()
    states, actions, rewards = [], [], []
    
    # Ensure initial state is a numpy array for consistent passing
    state = np.array(state) 

    while True:
        action = choose_action(state)
        # env.step() returns (next_state, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store data from this step
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = np.array(next_state) # Update state
        
        if done:
            # Prepare data types for training function
            # State: Convert list of arrays to 2D stacked array
            stacked_states = np.vstack(states).astype(np.float32)
            # Actions & Rewards: Convert lists to 1D NumPy arrays
            action_array = np.array(actions)
            reward_array = np.array(rewards)

            # Call training function with correctly formatted data
            train_on_episodes(stacked_states, action_array, reward_array)

            print(f"Episode: {episode + 1}, Total Rewards: {sum(rewards):.2f}")
            break

env.close()