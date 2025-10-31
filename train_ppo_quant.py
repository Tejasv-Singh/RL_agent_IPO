import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_quant import QuantIPOEnv

def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Quant IPO Allocation")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Number of training timesteps")
    parser.add_argument("--M", type=int, default=8, help="Number of IPOs (env arms)")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Initial capital")
    parser.add_argument("--feature_dim", type=int, default=4, help="IPO feature dimension")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="logs/ppo_quant", help="Model/log save directory")
    parser.add_argument("--delayed_reward", type=bool, default=False, help="Enable delayed (realized) reward mode")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # Env factory to support VecNormalize
    def make_env():
        env = QuantIPOEnv(M=args.M, feature_dim=args.feature_dim, initial_capital=args.capital,
                          delayed_reward=args.delayed_reward, seed=args.seed)
        env.seed(args.seed)
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    policy_kwargs = dict(net_arch=[128, 128])
    model = PPO('MlpPolicy', env,
                learning_rate=3e-4,
                batch_size=128,
                n_steps=2048,
                verbose=1,
                seed=args.seed,
                policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(args.save_dir, "ppo_quant"))
    env.save(os.path.join(args.save_dir, "vecnormalize.pkl"))
    print(f"Model and VecNormalize saved to {args.save_dir}")
if __name__ == "__main__":
    main()
