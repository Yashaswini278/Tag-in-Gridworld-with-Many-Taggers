import numpy as np

class GridWorldEnv:
    def __init__(self, n=15, k=2):
        self.n = n  # Grid size
        self.k = k  # Number of taggers
        
        self.action_space = 8  # 8 directions for runner
        self.action_labels = ['North', 'North-East', 'East', 'South-East', 'South', 'South-West', 'West', 'North-West']
        
        self.runner_pos = None
        self.tagger_positions = None
        self.steps = 0
        self.max_steps = 200  # Maximum steps per episode
        
        self._last_reward = 0  # Store last reward
    
    def reset(self):
        # Randomly initialize runners positions
        self.runner_pos = np.random.randint(0, self.n, size=2) 
        # Randomly initialize taggers positions 
        self.tagger_positions = np.random.randint(0, self.n, size=(self.k, 2)) 
        # Track number of steps 
        self.steps = 0
        return self._get_obs()
    
    def step(self, action):
        self.steps += 1
        
        # Move runner
        self._move_runner(action)
        
        # Move taggers (random coded)
        self._move_taggers()
        
        # Check if runner is caught
        caught = any(np.all(self.runner_pos == tagger_pos) for tagger_pos in self.tagger_positions)
        
        # Calculate reward
        if caught:
            reward = -100
            done = True
        elif self.steps >= self.max_steps:
            reward = 100
            done = True
        else:
            reward = 1  # Small positive reward for surviving
            done = False
        
        self._last_reward = reward  # Store reward for rendering
        return self._get_obs(), reward, done
    
    # Returns current runner and tagger positions 
    def _get_obs(self):
        return np.concatenate([self.runner_pos] + [tagger_pos for tagger_pos in self.tagger_positions])
    
    def _move_runner(self, action):
        # 8 directions: [N, NE, E, SE, S, SW, W, NW] - 2 steps
        directions = [(-2,0), (-2,2), (0,2), (2,2), (2,0), (2,-2), (0,-2), (-2,-2)]
        move = directions[action]
        self.runner_pos = np.clip(self.runner_pos + move, 0, self.n - 1) # dont move if <0 or >=n 

    
    def _move_taggers(self):
        # 4 directions: [N, E, S, W]
        directions = [(-1,0), (0,1), (1,0), (0,-1)]
        # move all the taggers randomly 
        for i in range(self.k):
            move = directions[np.random.randint(4)]
            self.tagger_positions[i] = np.clip(self.tagger_positions[i] + move, 0, self.n - 1)
    
    # used for simulating environment 
    def sample_action(self):
        action = np.random.randint(self.action_space)
        label = self.action_labels[action]
        return action, label

    def get_action_space(self):
        return self.action_space