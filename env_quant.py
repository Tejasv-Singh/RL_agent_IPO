import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional, Any
import numpy as np
from data_simulator_quant import IPODataSimulator

class QuantIPOEnv(gym.Env):
    """
    RL Environment for allocating fixed capital across M IPOs using a simulated market.
    Action: vector of allocation fractions (sum<=1), rescaled if sum>1.
    Observation: concatenated IPO features (M x feature_dim) + capital scalar.
    Reward: expected or delayed realized IPO profits minus risk penalty.
    """
    metadata = {"render_modes": [None]}
    
    def __init__(self, M: int = 8, feature_dim: int = 4, initial_capital: float = 1_000_000,
                 lambda_risk: float = 1.0, delayed_reward: bool = False, 
                 real_data_source: str = "mock_real", csv_path: Optional[str] = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.M = M
        self.feature_dim = feature_dim
        self.lambda_risk = lambda_risk
        self.initial_capital = initial_capital
        self.capital = float(initial_capital)
        self.delayed_reward = delayed_reward
        self.real_data_source = real_data_source
        self.csv_path = csv_path
        
        # For now, still use simulator (will integrate real data loader later)
        self.sim = IPODataSimulator(M, feature_dim, seed=seed)
        self.seed_value = seed
        self.pending = []  # Holds pending allocations for delayed reward
        self.time = 0
        
        # Observation: all IPO features flatten + [capital]
        obs_dim = M * feature_dim + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(M,), dtype=np.float32)
        
        # IPO listing delay and max steps
        self.ipo_listing_delay = 2
        self.max_steps = 40
        self.current_features = None
        self.current_exp_gain = None
        self.current_vol_20d = None

    def seed(self, seed: int = 0):
        """Seed the environment RNG."""
        self.sim.seed(seed)
        np.random.seed(seed)
        return [seed]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return initial observation."""
        if seed is not None:
            self.seed(seed)
        
        self.capital = float(self.initial_capital)
        self.pending = []
        self.time = 0
        
        # Sample initial batch of IPO data
        features, exp_gain, vol_20d = self.sim.sample_batch(batch_size=1)
        self.current_features = features[0]  # (M, feature_dim)
        self.current_exp_gain = exp_gain[0]  # (M,)
        self.current_vol_20d = vol_20d[0]    # (M,)
        
        obs = self._obs()
        return obs, {}

    def _obs(self) -> np.ndarray:
        """Construct observation: flattened IPO features + capital."""
        flat_features = self.current_features.flatten()  # (M * feature_dim,)
        cap_normalized = self.capital / self.initial_capital
        obs = np.concatenate([flat_features, [cap_normalized]], axis=0).astype(np.float32)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step of environment."""
        self.time += 1
        
        # Normalize action (allocations)
        action = np.clip(action, 0.0, 1.0)
        action_sum = np.sum(action)
        if action_sum > 1.0:
            action = action / action_sum
        
        # Allocate capital
        amounts_allocated = action * self.capital
        
        # Sample new IPO batch
        features, exp_gain, vol_20d = self.sim.sample_batch(batch_size=1)
        self.current_features = features[0]
        self.current_exp_gain = exp_gain[0]
        self.current_vol_20d = vol_20d[0]
        
        # Compute immediate reward (expected gains)
        immediate_gain = np.sum(amounts_allocated * self.current_exp_gain)
        
        # Risk penalty
        risk_penalty = self.lambda_risk * np.sum(amounts_allocated * self.current_vol_20d)
        
        reward = immediate_gain - risk_penalty
        
        # Update capital (simplified: gains are realized immediately in this version)
        self.capital += reward
        
        # Check terminal condition
        done = self.time >= self.max_steps or self.capital <= 0
        truncated = False
        
        obs = self._obs()
        info = {
            "time": self.time,
            "capital": self.capital,
            "allocated": np.sum(amounts_allocated),
            "reward_breakdown": {"gain": immediate_gain, "risk_penalty": risk_penalty}
        }
        
        return obs, float(reward), done, truncated, info

    def render(self, mode: str = 'human'):
        """Render environment (not implemented)."""
        pass
