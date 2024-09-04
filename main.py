import argparse

from environment import GridWorldEnv
from value_iteration import *
from temporal_difference import *
from visualize_env import visualize_grid_world, visualize_policy 
#from run import run_episodes, plot_average_rewards

def main(args):
    # Initialize environment
    env = GridWorldEnv()
    state = env.reset()

    if args.experiment == "visualize":
        # Visualize and simulate the environment
        visualize_grid_world(env, title="Initial State")
        for _ in range(args.steps):
            action, label = env.sample_action()
            state, reward, done = env.step(action)
            visualize_grid_world(env, title=f"After Action {label}")
            if done:
                break
    
    if args.experiment == "vi": 
        env = GridWorldEnv()
        env.reset()

        # Run value iteration algorithm
        print("Running Value Iteration...")
        vi_value, vi_policy = value_iteration(env)

        # Visualize policy learned 
        visualize_policy(env, vi_policy)

        # Run across episodes and plot average rewards 
        print("Running Episodes...")
        rewards = run_episodes(env, vi_policy, num_episodes=1000)
        plot_average_rewards(rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridWorld Environment Simulation and Visualization")
    parser.add_argument("--experiment", type=str, choices=["visualize", "vi", "td_learning"], 
                        help="Experiment to run: visualize (with simulation), simulate, value_iteration, or td_learning")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to simulate")


    args = parser.parse_args()
    main(args)
