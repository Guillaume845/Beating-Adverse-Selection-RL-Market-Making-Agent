import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GlostenMilgromEnv(gym.Env):
    def __init__(self, mu=0.3, gamma=0.05, max_inventory=10):
        super().__init__()
        self.mu = mu
        self.gamma = gamma
        self.max_inventory = max_inventory
        
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.array([-self.max_inventory, -1]),
            high=np.array([self.max_inventory, 1]),
            dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = 0
        self.last_trade = 0
        self.time = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.inventory, self.last_trade], dtype=np.float32)

    def step(self, action):
        spread = action + 1.0  
        ask = spread / 2.0     
        bid = -spread / 2.0    
        
        true_shock = self.np_random.normal(0, 2.0)
        trade_happened = 0
        pnl = 0
        
        is_informed = self.np_random.random() < self.mu
        
        if is_informed:
            if true_shock > ask: 
                trade_happened = 1      
                pnl = ask - true_shock  
                self.inventory -= 1
            elif true_shock < bid: 
                trade_happened = -1     
                pnl = true_shock - bid  
                self.inventory += 1
        else:
            if self.np_random.random() < 0.5:
                trade_happened = 1
                pnl = ask               
                self.inventory -= 1
            else:
                trade_happened = -1
                pnl = -bid              
                self.inventory += 1
                
        inventory_risk = self.gamma * (self.inventory ** 2)
        reward = pnl - inventory_risk
        
        self.last_trade = trade_happened
        self.time += 1
        
        done = self.time >= 200
        truncated = False
        
        if abs(self.inventory) > self.max_inventory:
            reward -= 50 
            done = True
            
        return self._get_obs(), float(reward), done, truncated, {}