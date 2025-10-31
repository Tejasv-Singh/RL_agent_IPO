import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_quant import QuantIPOEnv
from utils_metrics import sharpe_ratio, max_drawdown, annualized_return

def main():
    parser = argparse.ArgumentParser(description="Backtest trained PPO quant agent on IPO environment.")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO model ZIP")
    parser.add_argument("--episodes", type=int, default=20, help="Number of backtest episodes")
    parser.add_argument("--M", type=int, default=8, help="Number of IPOs (arms)")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    parser.add_argument("--feature_dim", type=int, default=4, help="Feature dimension")
    parser.add_argument("--delayed_reward", type=bool, default=False, help="Delayed (realized) reward mode")
    parser.add_argument("--seed", type=int, default=123, help="Seed")
    parser.add_argument("--vecnormalize", type=str, default=None, help="Path to VecNormalize.pkl (optional)")
    parser.add_argument("--logdir", type=str, default="logs/backtest", help="PNG output directory")
    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    # Setup environment
    def make_env():
        env = QuantIPOEnv(M=args.M, feature_dim=args.feature_dim, initial_capital=args.capital, delayed_reward=args.delayed_reward, seed=args.seed)
        env.seed(args.seed)
        return env
    env = DummyVecEnv([make_env])
    vec_file = args.vecnormalize if args.vecnormalize else os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")
    if os.path.exists(vec_file):
        env = VecNormalize.load(vec_file, env)
        env.training = False
        env.norm_reward = False
    model = PPO.load(args.model)
    episode_profits = []
    all_wealth_paths = []
    print(f"Backtesting {args.episodes} episodes...")
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        wealth_path = [args.capital]
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, info = env.step(action)
            done = bool(done_vec[0])
            capital = float(obs[0, -1])
            wealth_path.append(capital)
        profit = wealth_path[-1] - args.capital
        episode_profits.append(profit)
        all_wealth_paths.append(wealth_path)
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
    print(f"Mean profit/episode: {mean_profit:.1f}, Std: {std_profit:.1f}, Sharpe: {sharpe:.2f}, Max Drawdown: {mdd:.2%}, Ann. Ret: {ann_ret:.2%}")
    plt.figure(figsize=(8,4))
    plt.plot(avg_wealth)
    plt.title(f"Average Wealth Path ({args.episodes} eps), Mean PnL: {mean_profit:.0f}")
    plt.xlabel("Step")
    plt.ylabel("Wealth ($)")
    plt.tight_layout()
    png_path = os.path.join(args.logdir, "avg_wealth.png")
    plt.savefig(png_path)
    print(f"PNG saved: {png_path}")
if __name__ == "__main__":
    main()
