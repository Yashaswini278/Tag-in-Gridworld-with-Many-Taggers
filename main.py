import argparse

from environment import GridWorldEnv
from value_iteration import *
from temporal_difference import *
from visualize_env import visualize_grid_world, visualize_policy 
from run import run_episodes, plot_average_rewards

def main(args):
    # Initialize environment
    env = GridWorldEnv(args.n, args.k)
    state = env.reset()

    if args.exp == "visualize":
        # Visualize and simulate the environment
        visualize_grid_world(env, title="Initial State")
        for _ in range(args.steps):
            action, label = env.sample_action()
            state, reward, done = env.step(action)
            visualize_grid_world(env, title=f"After Action {label}")
            if done:
                break
    
    if args.exp == "vi": 

        # Run value iteration algorithm
        print("Running Value Iteration...")
        vi_value, vi_initial_policy, vi_intermediate_policy, vi_policy = value_iteration(env, args.gamma, args.theta)

        # Visualize policy learned 
        visualize_policy(env, vi_initial_policy, 'Initial Policy VI')
        visualize_policy(env, vi_intermediate_policy, 'Intermediate Policy VI')
        visualize_policy(env, vi_policy, 'Final Policy VI')

    if args.exp == "td": 

        # Run value iteration algorithm
        print("Running Temporal Difference Learning...")
        td_value, td_initial_policy, td_intermediate_policy, td_policy = td_learning(env, args.episodes, args.alpha, args.gamma, args.epsilon, args.lamda)

        # Visualize policy learned 
        visualize_policy(env, td_initial_policy, 'Initial Policy TD Learning')
        visualize_policy(env, td_intermediate_policy, 'Intermediate Policy TD Learning')
        visualize_policy(env, td_policy, 'Final Policy TD Leanring')

    if args.exp == "compare": 
        env = GridWorldEnv()
        state = env.reset()

        # Value Iteration
        print("Running Value Iteration")
        vi_V, _, _, vi_policy = value_iteration(env)
        vi_rewards = run_episodes(env, vi_policy, args.numepisodes)
        
        # TD Learning
        print("Running Temporal Difference Learning")
        td_V, _, _, td_policy = td_learning(env, episodes=args.episodes)
        td_rewards = run_episodes(env, td_policy, args.numepisodes)

        plot_average_rewards(vi_rewards, td_rewards, args.numepisodes)

        visualize_policy(env, vi_policy, 'Policy VI')
        visualize_policy(env, td_policy, 'Policy TD Learning')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridWorld Environment Simulation and Visualization")

    parser.add_argument("--n", type=int, default=15, help="grid size")
    parser.add_argument("--k", type=int, default=2, help="number of taggers")

    parser.add_argument("--exp", type=str, choices=["visualize", "vi", "td", "compare"], 
                        help="Experiment to run: visualize (with simulation), value_iteration, or td_learning")
    
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to simulate")

    parser.add_argument("--gamma", type=float, default=0.99, help="gamma for value iteration or td learning")
    parser.add_argument("--theta", type=float, default=1e-8, help="theta for value iteration")

    parser.add_argument("--episodes", type=int, default=100, help="number of episodes for td learning")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha for td learning")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon for td learning")
    parser.add_argument("--lamda", type=float, default=0.9, help="lamda for td learning")

    parser.add_argument("--numepisodes", type=int, default=200, help="number of episodes for comparing the algorithms")

    args = parser.parse_args()
    main(args)
