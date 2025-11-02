import argparse
import os
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_quant import QuantIPOEnv
import random

def main():
    seed_val = random.randint(0, 1_000_000)
    parser = argparse.ArgumentParser(description="Train RL agent for Quant IPO Allocation")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Number of training timesteps")
    parser.add_argument("--M", type=int, default=8, help="Number of IPOs (env arms)")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    parser.add_argument("--feature_dim", type=int, default=4, help="IPO feature dimension")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="logs/ppo_quant", help="Model/log save directory")
    parser.add_argument("--delayed_reward", type=bool, default=False, help="Enable delayed (realized) reward mode")
    # ADD THESE NEW ARGUMENTS
    parser.add_argument("--data_source", type=str, default="mock_real", 
                        help="'mock_real', 'csv', or 'yfinance'")
    parser.add_argument("--csv_path", type=str, default=None, 
                        help="Path to IPO CSV file (required if data_source='csv')")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Env factory to support VecNormalize
    def make_env():
        env = QuantIPOEnv(M=args.M, feature_dim=args.feature_dim, initial_capital=args.capital,
                          delayed_reward=args.delayed_reward, 
                          real_data_source=args.data_source,
                          csv_path=args.csv_path,
                          seed=seed_val)
        env.seed(seed_val)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    policy_kwargs = dict(net_arch=[128, 128])
    model = PPO('MlpPolicy', env,
                learning_rate=3e-4,
                batch_size=128,
                n_steps=2048,
                verbose=1,
                seed=args.seed,
                policy_kwargs=policy_kwargs)
    
    print(f"Starting training with seed: {seed_val}")
    model.learn(total_timesteps=args.timesteps)
    
    # Save model
    model.save(os.path.join(args.save_dir, "ppo_quant"))
    env.save(os.path.join(args.save_dir, "vecnormalize.pkl"))
    
    # Save config for later use
    config = {
        "M": args.M,
        "feature_dim": args.feature_dim,
        "capital": args.capital,
        "data_source": args.data_source,
        "csv_path": args.csv_path,
        "delayed_reward": args.delayed_reward,
    }
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Model and VecNormalize saved to {args.save_dir}")
    print(f"Config saved to {os.path.join(args.save_dir, 'config.json')}")

if __name__ == "__main__":
    main()
