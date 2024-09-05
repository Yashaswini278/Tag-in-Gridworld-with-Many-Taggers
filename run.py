import numpy as np
import matplotlib.pyplot as plt 

def run_episodes(env, policy, num_episodes=1000):
    rewards = []
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        done = False
        while not done:
            runner_pos = env.runner_pos
            action = policy[runner_pos[0], runner_pos[1]]
            _, reward, done = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return rewards

def plot_average_rewards(rewards, num_episodes=1000, title = None):
    cumulative_rewards = np.cumsum(rewards)
    average_rewards = cumulative_rewards / (np.arange(num_episodes) + 1)
    plt.plot(average_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.show()
