# evaluate_utils.py
import numpy as np
import json
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_quant import QuantIPOEnv
from utils_metrics import sharpe_ratio, max_drawdown, annualized_return
import random

def accuracy_score_from_wealth(wealth):
    # wealth: 1D numpy array
    returns = np.diff(wealth) / (wealth[:-1] + 1e-8)
    sharpe = sharpe_ratio(returns)
    mdd = max_drawdown(wealth)
    ann = annualized_return(wealth)
    acc = sharpe * (1 - mdd) * (1 + ann)
    acc = max(0.0, acc)  # floor at 0
    return acc, sharpe, mdd, ann

def make_env_factory(M, feature_dim, capital, delayed_reward):
    def _make():
        seed_val = random.randint(0, 999999)
        env = QuantIPOEnv(M=M, feature_dim=feature_dim, initial_capital=capital,
                          delayed_reward=delayed_reward, seed=seed_val)
        env.seed(seed_val)
        return env
    return _make

def evaluate_model(model_path, M, feature_dim, capital, episodes=20, vecnormalize_path=None, delayed_reward=False, seed=None):
    """
    Loads a model and returns aggregated metrics and per-episode wealth paths.
    Returns dict with accuracy_list, sharpe_list, mdd_list, ann_list, profits_list, avg_wealth
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = DummyVecEnv([make_env_factory(M, feature_dim, capital, delayed_reward)])
    # try load normalization if given
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False

    if not model_path.endswith(".zip"):
        model_path = model_path + ".zip"
    model = PPO.load(model_path)

    accs, sharps, mdds, anns, profits = [], [], [], [], []
    all_paths = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        wealth = [capital]
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done_vec, info = env.step(action)
            done = bool(done_vec[0])
            capital_now = float(obs[0, -1])
            wealth.append(capital_now)
        wealth = np.array(wealth)
        acc, sharpe, mdd, ann = accuracy_score_from_wealth(wealth)
        profit = wealth[-1] - wealth[0]
        accs.append(acc)
        sharps.append(sharpe)
        mdds.append(mdd)
        anns.append(ann)
        profits.append(profit)
        all_paths.append(wealth)

    return {
        "accuracy_list": np.array(accs),
        "sharpe_list": np.array(sharps),
        "mdd_list": np.array(mdds),
        "ann_list": np.array(anns),
        "profits": np.array(profits),
        "all_paths": all_paths
    }
