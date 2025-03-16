import numpy as np
import gymnasium as gym
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def initialize_environment():
    """
    Initializes the Taxi-v3 environment and Q-table.
    Returns:
        env: The gymnasium environment.
        q_table: Initialized Q-table.
    """
    env = gym.make("Taxi-v3")
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    return env, q_table

def train_q_learning(env, q_table, alpha=0.1, gamma=0.6, epsilon=0.1, episodes=1000):
    """
    Trains the agent using Q-learning.
    """
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _, _ = env.step(action)
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            state = next_state
    logging.info("Training finished.")

def evaluate_q_learning(env, q_table, episodes=100):
    """
    Evaluates the trained Q-learning agent.
    """
    total_epochs, total_penalties = 0, 0
    for _ in range(episodes):
        state, _ = env.reset()
        epochs, penalties = 0, 0
        done = False
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, _, _ = env.step(action)
            if reward == -10:
                penalties += 1
            epochs += 1
        total_penalties += penalties
        total_epochs += epochs
    logging.info(f"Results after {episodes} episodes:")
    logging.info(f"Average timesteps per episode: {total_epochs / episodes:.2f}")
    logging.info(f"Average penalties per episode: {total_penalties / episodes:.2f}")

def main():
    """
    Main function to initialize, train, and evaluate the Q-learning model.
    """
    env, q_table = initialize_environment()
    train_q_learning(env, q_table)
    evaluate_q_learning(env, q_table)

if __name__ == "__main__":
    main()
