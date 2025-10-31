import numpy as np
import pytest
from data_simulator_quant import IPODataSimulator
from env_quant import QuantIPOEnv

def test_ipo_simulator_shapes():
    sim = IPODataSimulator(M=5, feature_dim=4, seed=42)
    features, exp_gain, vol_20d = sim.sample_batch(3)
    assert features.shape == (3, 5, 4)
    assert exp_gain.shape == (3, 5)
    assert vol_20d.shape == (3, 5)
    actual_gains = sim.sample_realized_listing_gain(2)
    assert actual_gains.shape == (2, 5)
    allot = sim.sample_allotment_fraction(2)
    assert allot.shape == (2, 5)
    assert np.all((0.3 <= allot) & (allot <= 1.0))
def test_env_step_expected_mode():
    env = QuantIPOEnv(M=4, feature_dim=3, initial_capital=100_000, delayed_reward=False, seed=7)
    obs, _ = env.reset(seed=7)
    init_cap = obs[-1]
    for i in range(5):
        action = np.ones(4) / 4  # Equal allocation
        obs2, reward, done, truncated, info = env.step(action)
        # Capital should update (simulate full liquidity)
        assert isinstance(obs2, np.ndarray)
        assert isinstance(reward, float)
        assert obs2[-1] >= 0.0
        obs = obs2
def test_env_step_delayed_mode():
    env = QuantIPOEnv(M=3, feature_dim=3, initial_capital=50_000, delayed_reward=True, seed=21)
    obs, _ = env.reset(seed=21)
    for i in range(6):
        action = np.ones(3) / 3
        obs2, reward, done, truncated, info = env.step(action)
        assert isinstance(obs2, np.ndarray)
        assert isinstance(reward, float)
        obs = obs2
