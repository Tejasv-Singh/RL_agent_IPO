import argparse
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_quant import QuantIPOEnv
from utils_metrics import sharpe_ratio, max_drawdown, annualized_return

def main():
    parser = argparse.ArgumentParser(description="Backtest trained PPO quant agent on IPO environment.")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO model (without .zip)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--delayed_reward", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vecnormalize", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="logs/backtest")
    parser.add_argument("--data_source", type=str, default="mock_real", 
                        help="'mock_real', 'csv', or 'yfinance'")
    parser.add_argument("--csv_path", type=str, default=None, 
                        help="Path to IPO CSV file (required if data_source='csv')")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    # --- Auto-load training config if exists ---
    config_path = os.path.join(os.path.dirname(args.model), "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            args.M = args.M or cfg.get("M", 5)
            args.feature_dim = args.feature_dim or cfg.get("feature_dim", 4)
            args.capital = args.capital or cfg.get("capital", 100000)
            args.data_source = cfg.get("data_source", args.data_source)
            args.csv_path = cfg.get("csv_path", args.csv_path)
            print(f"Loaded training config: M={args.M}, feature_dim={args.feature_dim}, capital={args.capital}, data_source={args.data_source}")
    else:
        print("⚠️ No config.json found, using command-line defaults.")

    # --- Randomize seed for unique runs ---
    args.seed = args.seed or random.randint(0, 1_000_000)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"🎲 Using random seed: {args.seed}")

    # --- Create environment factory ---
    def make_env():
        env = QuantIPOEnv(
            M=args.M, feature_dim=args.feature_dim,
            initial_capital=args.capital,
            delayed_reward=args.delayed_reward,
            real_data_source=args.data_source,
            csv_path=args.csv_path,
            seed=random.randint(0, 1_000_000)
        )
        env.seed(args.seed)
        return env

    env = DummyVecEnv([make_env])

    # --- Load VecNormalize (if available) ---
    vec_file = args.vecnormalize or os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")
    if os.path.exists(vec_file):
        env = VecNormalize.load(vec_file, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize from {vec_file}")
    else:
        print("⚠️ VecNormalize not found; running without normalization.")

    # --- Load PPO model ---
    model_path = args.model if args.model.endswith(".zip") else args.model + ".zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = PPO.load(model_path)
    print(f"Model loaded from {model_path}")

    # --- Backtest ---
    episode_profits = []
    all_wealth_paths = []
    print(f"Backtesting {args.episodes} episodes...\n")

    for ep in range(args.episodes):
        obs = env.reset()
        done = [False]  # DummyVecEnv returns list
        wealth_path = [args.capital]
        
        while not done[0]:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            # Convert numpy array to scalar
            reward_scalar = float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            wealth_path.append(wealth_path[-1] + reward_scalar)
        
        profit = wealth_path[-1] - args.capital
        episode_profits.append(profit)
        all_wealth_paths.append(wealth_path)
        print(f"Episode {ep+1}/{args.episodes}: Profit = ${profit:.2f}")

    # --- Metrics ---
    profits = np.array(episode_profits)
    mean_profit = float(np.mean(profits))
    std_profit = float(np.std(profits))
    min_len = min(map(len, all_wealth_paths))
    wealth_arr = np.array([wp[:min_len] for wp in all_wealth_paths])
    avg_wealth = np.mean(wealth_arr, axis=0)
    avg_returns = np.diff(avg_wealth) / (avg_wealth[:-1] + 1e-8)

    sharpe = sharpe_ratio(avg_returns)
    mdd = max_drawdown(avg_wealth)
    ann_ret = annualized_return(avg_wealth)

    print(f"\n📊 Mean Profit/Ep: ${mean_profit:.1f}, Std: ${std_profit:.1f}")
    print(f"📈 Sharpe: {sharpe:.2f}, Max DD: {mdd:.2%}, Annualized Return: {ann_ret:.2%}")

    # --- Plot ---
    plt.figure(figsize=(8, 4))
    plt.plot(avg_wealth, label="Average Wealth")
    plt.title(f"Average Wealth Path ({args.episodes} eps), Mean PnL: ${mean_profit:.0f}")
    plt.xlabel("Step")
    plt.ylabel("Wealth ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(args.logdir, f"avg_wealth_seed{args.seed}.png")
    plt.savefig(png_path, dpi=100)
    print(f"\n📊 PNG saved: {png_path}")
    plt.close()

if __name__ == "__main__":
    main()
