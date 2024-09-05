import numpy as np
import matplotlib.pyplot as plt
from environment import GridWorldEnv
from visualize_env import visualize_policy

def td_learning(env, episodes=500, alpha=0.3, gamma=0.99, epsilon=0.1, lambd=0.8):
    # Initialize state-value function V(s)
    V = np.zeros((env.n, env.n))  
    
    for T in range(episodes):
        # Initialize eligibility trace E(s) for each state
        E = np.zeros((env.n, env.n))  
        # V_T(s) for the current episode is initialized as V_T-1(s)
        V_T = V.copy()
        # Start new episode
        state = env.reset()  
        done = False
        
        while not done:
            if np.random.random() < epsilon:
                # Exploration: choose a random action
                action, label = env.sample_action()  
            else:
                # Choose the action with the highest state-value
                action = np.argmax([V_T[state[0], state[1]] + gamma * V_T[state[0], state[1]]])

            # Get next state and reward
            next_state, reward, done = env.step(action)  
            
            # Increment eligibility for the current state
            E[state[0], state[1]] += 1  
            
            # Calculate TD error
            td_error = reward + gamma * V[next_state[0], next_state[1]] - V[state[0], state[1]]
            
            # Update the value function for all states
            for i in range(env.n):
                for j in range(env.n):
                    V_T[i, j] += alpha * td_error * E[i, j]
                    # Decay the eligibility trace
                    E[i, j] *= lambd  
            
            # Move to the next state
            state = next_state  

        # Update V for the next episode 
        V = V_T  

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

