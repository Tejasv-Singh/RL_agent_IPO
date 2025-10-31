# Quant IPO RL Agent: Portfolio Allocation for Simulated IPOs

## Project Overview
This repo provides a reproducible, quant-focused RL agent to allocate fixed capital across multiple simulated IPO opportunities. It features a realistic IPO + market simulator and a Gym environment compatible with Stable-Baselines3 (PPO), supporting both immediate expected and delayed realized reward modes. The codebase is fully self-contained, requires no external data, and enables vectorized training/backtesting for robust financial RL experimentation.

## Features
- Simulated quant IPO + market environment (no external data needed)
- Reproducible randomness with explicit seeding (`numpy.random.RandomState`)
- Two reward modes: immediate expected vs. delayed realized
- Gym v0.26.2, Stable-Baselines3, VecNormalize-compatible
- Utilities: Sharpe ratio, annualized return, max drawdown calculation
- Training and backtesting scripts with CLI and plotting
- Unit tests for critical functionality (simulator & env)

## Reward Modes Explained
- **Immediate Expected Reward (default):** Agent gets expected profit reward instantly based on allocation, expected listing gain, and risk penalty. Allows smoother gradients and fast credit assignment.
- **Delayed Realized Reward (`--delayed_reward True`):** Rewards are only granted after IPOs actually list. Allocated capital is reserved, and realized profit/loss is returned when IPOs list. Rewards per step are mostly zero until listing occurs (except time-penalty), better modeling real-world delay and risk.
- **Toggling:** Use the `--delayed_reward` CLI flag in training/backtesting scripts.

## Project Structure
- `README.md`: This file. Project details, install, usage, reward modes.
- `requirements.txt`: Python dependencies (Python 3.9+).
- `data_simulator_quant.py`: Quantitative IPO data simulator. Provides `IPODataSimulator` class.
- `env_quant.py`: Gym environment for RL allocation (`QuantIPOEnv`).
- `train_ppo_quant.py`: Train RL agent via PPO (Stable-Baselines3) with config options.
- `backtest_quant.py`: Backtests saved models, computes and plots metrics.
- `utils_metrics.py`: Helpers for Sharpe ratio, drawdown, returns.
- `tests/test_simulator_env.py`: Unit tests (pytest compatible).
- `run.sh`: Convenience script for full repo demo.
- `.gitignore`: Ignore venv, logs, pyc, and typical build files.

## Installation & Quick Start
```bash
# Clone repo and enter project directory
cd "RL agent"
# (Recommended) create venv
python -m venv venv && source venv/bin/activate
# Install Python dependencies
pip install -r requirements.txt
```

## Training Example
```bash
python train_ppo_quant.py --timesteps 200000 --M 8 --capital 1000000 --seed 0 --save-dir logs/ppo_quant
```

## Backtest Example
```bash
python backtest_quant.py --model logs/ppo_quant/ppo_quant.zip --episodes 50
```

## Run All: install, train short model, backtest
```bash
bash run.sh
```

## File List & Brief Descriptions
| File                  | Description                                             |
|-----------------------|---------------------------------------------------------|
| README.md             | Overview, usage, reward mode tradeoff                   |
| requirements.txt      | Package dependencies                                    |
| data_simulator_quant.py | Quant IPO/market simulator  (`IPODataSimulator`)       |
| env_quant.py          | Gym RL environment (`QuantIPOEnv`)                      |
| train_ppo_quant.py    | Script to train RL agent                                |
| backtest_quant.py     | Analyze trained agent, metrics & plot                   |
| utils_metrics.py      | Financial metrics utilities                             |
| tests/test_simulator_env.py | Unit tests for critical components             |
| run.sh                | Single-command install, train, backtest demo            |
| .gitignore            | Ignore venv, logs, cache, build artifacts, pyc          |

## License
MIT License (see LICENSE file if present)
