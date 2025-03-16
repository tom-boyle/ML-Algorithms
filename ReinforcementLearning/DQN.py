import gymnasium as gym
import numpy as np
import tensorflow as tf
import logging
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize the gymnasium environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.95  # Discount factor
EPSILON = 1.0  # Exploration rate
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
BATCH_SIZE = 64
MEMORY = deque(maxlen=2000)


def build_dqn_model():
    """
    Builds and compiles a Deep Q-Network (DQN) model.
    
    Returns:
        model (Sequential): Compiled DQN model.
    """
    model = Sequential([
        Flatten(input_shape=(1, state_size)),
        Dense(24, activation="relu"),
        Dense(24, activation="relu"),
        Dense(action_size, activation="linear")
    ])
    model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
    return model


# Initialize the DQN model
model = build_dqn_model()


def replay(memory, batch_size):
    """
    Trains the DQN model using experience replay.

    Args:
        memory (deque): Replay memory storing past experiences.
        batch_size (int): Number of samples to use for training.
    """
    minibatch = random.sample(memory, batch_size)
    
    for state, action, reward, next_state, done in minibatch:
        target = reward if done else (reward + GAMMA * np.amax(model.predict(next_state, verbose=0)[0]))
        target_f = model.predict(state, verbose=0)
        target_f[0][action] = target
        
        model.fit(state, target_f, epochs=1, verbose=0)


def train_dqn(episodes=1000, max_timesteps=500):
    """
    Trains the DQN model on the CartPole environment.

    Args:
        episodes (int): Number of training episodes.
        max_timesteps (int): Maximum timesteps per episode.
    """
    global EPSILON  # Allow modification of global epsilon

    for episode in range(episodes):
        state = env.reset()[0]
        state = np.reshape(state, [1, 1, state_size])

        for timestep in range(max_timesteps):
            if np.random.rand() <= EPSILON:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(model.predict(state, verbose=0)[0])  # Exploit

            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -10  # Penalize game over

            next_state = np.reshape(next_state, [1, 1, state_size])
            MEMORY.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                logging.info(f"Episode: {episode}/{episodes}, Score: {timestep}, Epsilon: {EPSILON:.2f}")
                break

            if len(MEMORY) > BATCH_SIZE:
                replay(MEMORY, BATCH_SIZE)

        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY


# Train the model
train_dqn()
