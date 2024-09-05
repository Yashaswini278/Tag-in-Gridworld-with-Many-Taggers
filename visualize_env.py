import matplotlib.pyplot as plt
import numpy as np
from environment import GridWorldEnv

def visualize_grid_world(env, title=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid
    for i in range(env.n + 1):
        ax.axhline(i, color='lightgray', linewidth=1)
        ax.axvline(i, color='lightgray', linewidth=1)
    
    # Draw runner
    runner_x, runner_y = env.runner_pos
    ax.add_artist(plt.Circle((runner_x + 0.5, runner_y + 0.5), 0.4, color='blue', zorder=2))
    ax.text(runner_x + 0.5, env.n - runner_y - 0.5, 'R', ha='center', va='center', color='white', fontweight='bold', zorder=3)
    
    # Draw taggers
    for i, (tagger_y, tagger_x) in enumerate(env.tagger_positions):
        ax.add_artist(plt.Circle((tagger_x + 0.5, env.n - tagger_y - 0.5), 0.3, color='red', zorder=2))
        ax.text(tagger_x + 0.5, env.n - tagger_y - 0.5, f'T{i+1}', ha='center', va='center', color='white', fontweight='bold', zorder=3)
    
    # Set plot limits and labels
    ax.set_xlim(0, env.n)
    ax.set_ylim(0, env.n)
    ax.set_xticks(np.arange(0.5, env.n, 1))
    ax.set_yticks(np.arange(0.5, env.n, 1))
    ax.set_xticklabels(range(env.n))
    ax.set_yticklabels(range(env.n))
    
    # Add title and additional information
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    info_text = f"Steps: {env.steps}/{env.max_steps}\nLast Reward: {env._last_reward}"
    ax.text(env.n + 0.5, env.n, info_text, ha='left', va='top', fontsize=12)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Runner', markerfacecolor='blue', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Tagger', markerfacecolor='red', markersize=15)
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.tight_layout()
    plt.show()

def visualize_policy(env, policy, title="Policy Visualization"):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw grid
    for i in range(env.n + 1):
        ax.axhline(i, color='lightgray', linewidth=1)
        ax.axvline(i, color='lightgray', linewidth=1)
    
    # Define action directions
    directions = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
    
    # Draw policy arrows
    for i in range(env.n):
        for j in range(env.n):
            action = policy[i, j]
            dy, dx = directions[action]
            ax.arrow(j + 0.5, env.n - i - 0.5, dx * 0.4, dy * 0.4, 
                     head_width=0.2, head_length=0.2, fc='k', ec='k', zorder=2)
    
    # Set plot limits and labels
    ax.set_xlim(0, env.n)
    ax.set_ylim(0, env.n)
    ax.set_xticks(np.arange(0.5, env.n, 1))
    ax.set_yticks(np.arange(0.5, env.n, 1))
    ax.set_xticklabels(range(env.n))
    ax.set_yticklabels(range(env.n-1, -1, -1))
    
    # Add title
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='>', color='k', label='Action', markersize=15, linestyle='None')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
    
    plt.tight_layout()
    plt.show()