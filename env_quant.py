import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict, Optional, Any
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
                 lambda_risk: float = 1.0, delayed_reward: bool = False, seed: Optional[int] = None):
        super().__init__()
        self.M = M
        self.feature_dim = feature_dim
        self.lambda_risk = lambda_risk
        self.initial_capital = initial_capital
        self.capital = float(initial_capital)
        self.delayed_reward = delayed_reward
        self.sim = IPODataSimulator(M, feature_dim, seed=seed)
        self.seed_value = seed
        self.pending = []  # Holds pending allocations for delayed reward
        self.time = 0
        # Observation: all IPO features flatten + [capital]
        obs_dim = M * feature_dim + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # Action: allocation fractions, shape: (M,)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(M,), dtype=np.float32)
        # IPO list timing for delayed reward:
        self.ipo_listing_delay = 2  # Steps from allocation to listing
        self.max_steps = 40
    def seed(self, seed: int = 0):
        """Seed the environment and underlying simulator."""
        self.seed_value = seed
        self.sim.seed(seed)
        np.random.seed(seed)
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.seed(seed)
        self.capital = float(self.initial_capital)
        self.pending = []  # (step_allocated, alloc_amt (M,), allot_draw (M,), gain_draw (M,))
        self.time = 0
        features, exp_gain, vol_20d = self.sim.sample_batch(1)
        self.cur_features = features[0]
        self.cur_exp_gain = exp_gain[0]
        self.cur_vol_20d = vol_20d[0]
        obs = self._obs()
        return obs, {}
    def _obs(self) -> np.ndarray:
        # Flatten IPO features and append available capital scalar
        return np.concatenate([self.cur_features.flatten(), [self.capital]], dtype=np.float32)
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        done = False
        # Normalize and apply action (allocation fractions)
        action = np.clip(action, 0, 1)
        allocation = action / np.clip(action.sum(), 1e-8, 1.0)  # Renormalize if sum>1
        alloc_amt = self.capital * allocation
        # For risk penalty, use expected 20d volatility (annualized proxy)
        risk_penalty = self.lambda_risk * np.sum(alloc_amt * self.cur_vol_20d)
        reward = 0.0
        info = {}
        if self.delayed_reward:
            # Store allocation for pending listing
            allot_draw = self.sim.sample_allotment_fraction()[0]
            gain_draw = self.sim.sample_realized_listing_gain()[0]
            self.pending.append({
                'step': self.time,
                'alloc_amt': alloc_amt.copy(),
                'allot_draw': allot_draw.copy(),
                'gain_draw': gain_draw.copy()
            })
            # Remove invested portion from capital
            capital_allocated = alloc_amt.sum()
            self.capital -= capital_allocated
            # Add time-penalty (-0.001 per step as incentive to finish)
            reward = -0.001 * self.capital / self.initial_capital
            # Check matured IPO(s) to realize profit/loss
            matured = [x for x in self.pending if self.time - x['step'] >= self.ipo_listing_delay]
            for x in matured:
                realized_profit = np.sum(x['alloc_amt'] * x['allot_draw'] * x['gain_draw'])
                self.capital += np.sum(x['alloc_amt'] * x['allot_draw']) + realized_profit
                reward += realized_profit - np.sum(x['alloc_amt'] * self.cur_vol_20d) * self.lambda_risk
                self.pending.remove(x)
        else:
            # Immediate expected reward: reward = expected profit minus risk penalty
            expected_profit = np.sum(alloc_amt * self.cur_exp_gain)
            reward = expected_profit - risk_penalty
            self.capital -= alloc_amt.sum()  # Remove invested
            # Refund capital used (simulate full liquidity at each step)
            self.capital += alloc_amt.sum() + expected_profit
        # Next market state
        features, exp_gain, vol_20d = self.sim.sample_batch(1)
        self.cur_features = features[0]
        self.cur_exp_gain = exp_gain[0]
        self.cur_vol_20d = vol_20d[0]
        self.time += 1
        # Truncate episode?
        if self.time >= self.max_steps or self.capital <= 10:
            done = True
        obs = self._obs()
        truncated = False
        return obs, reward, done, truncated, info
    def render(self, mode: str = 'human'):
        print(f"Step: {self.time}, Capital: {self.capital:.2f}")
        print(f"Current IPO Features: {self.cur_features}")
    # Note: IPO allocation simplification as lot-size multiples can be documented, but is not enforced in code for vectorization.
