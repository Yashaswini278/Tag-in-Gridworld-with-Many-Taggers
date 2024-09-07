import numpy as np
import matplotlib.pyplot as plt 
from environment import GridWorldEnv

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

def plot_average_rewards(vi_rewards, td_rewards, num_episodes=1000, title=None):
    vi_cumulative_rewards = np.cumsum(vi_rewards)
    vi_average_rewards = vi_cumulative_rewards / (np.arange(num_episodes) + 1)
    
    td_cumulative_rewards = np.cumsum(td_rewards)
    td_average_rewards = td_cumulative_rewards / (np.arange(num_episodes) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(vi_average_rewards, label='Value Iteration')
    plt.plot(td_average_rewards, label='TD Learning')
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title(title or "Comparison of Value Iteration and TD Learning")
    plt.legend()
    plt.grid(True)
    plt.show()


