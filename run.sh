#!/usr/bin/env bash
set -e
DIR="$(dirname "$(realpath "$0")")"
echo "[1/4] Creating venv..."
python3 -m venv "$DIR/venv"
echo "[2/4] Activating venv & installing requirements..."
source "$DIR/venv/bin/activate"
pip install --upgrade pip
pip install -r "$DIR/requirements.txt"
echo "[3/4] Training PPO agent (short run)..."
python "$DIR/train_ppo_quant.py" --timesteps 15000 --M 5 --capital 100000 --seed 13 --save-dir "$DIR/logs/demo"
echo "[4/4] Backtesting PPO agent..."
python "$DIR/backtest_quant.py" --model "$DIR/logs/demo/ppo_quant.zip" --episodes 3 --M 5 --capital 100000 --seed 13 --logdir "$DIR/logs/demo"
echo "[Done] Demo run complete."
