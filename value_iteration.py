import numpy as np
import matplotlib.pyplot as plt
from environment import GridWorldEnv
from visualize_env import visualize_grid_world, visualize_policy

def value_iteration(env, gamma=0.99, theta=1e-8):
    # we choose some epsilon such that 
    # theta = eps^((1-gamaa)/2*gamma) -> used for stopping criteria
    # initialize value matrix for each state (square of the grid)
    V = np.zeros((env.n, env.n))
    
    while True:
        delta = 0 # defined for stopping criteria
        for i in range(env.n):
            for j in range(env.n):
                v = V[i, j]
                action_values = []
                for action in range(env.get_action_space()):
                    # start runner pos from current state
                    env.runner_pos = np.array([i, j])
                    # step the environment -> get next state, reward, done (caught or reached max steps => true)
                    next_state, reward, done = env.step(action)
                    # update based on Bellman Optimality equations
                    action_values.append(reward + gamma * V[next_state[0], next_state[1]])
                # choose max among the action values 
                V[i, j] = max(action_values)

                delta = max(delta, abs(v - V[i, j]))
        
        # stopping criteria 
        if delta < theta:
            break
    
    # choose policy based on the values calculated
    policy = np.zeros((env.n, env.n), dtype=int)
    initial_policy = policy.copy()
    for i in range(env.n):
        for j in range(env.n):
            action_values = []
            for action in range(env.get_action_space()):
                # same steps described in value calculation
                env.runner_pos = np.array([i, j])
                next_state, reward, done = env.step(action)
                action_values.append(reward + gamma * V[next_state[0], next_state[1]])
            # choose action with highest value 
            policy[i, j] = np.argmax(action_values)
            if(i == env.n//2 and j == env.n//2):
                intermediate_policy = policy.copy()
    
    return V, initial_policy, intermediate_policy, policy


