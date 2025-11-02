import streamlit as st
import subprocess
import os
import glob
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_quant import QuantIPOEnv

# ===== PAGE CONFIG =====
st.set_page_config(layout="wide", page_title="Quant IPO RL Dashboard")
st.title("Quant IPO RL Agent Dashboard")

# ===== SIDEBAR NAVIGATION =====
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Training", "Backtesting", "Allocation Advisor"])

if page == "Home":
    st.header("Welcome to IPO Allocation AI")
    st.write("""
    This dashboard helps you:
    - Explore real IPO data
    - Train a PPO reinforcement learning agent
    - Backtest trained models
    - Get AI-powered IPO allocation recommendations
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Models", len(glob.glob("logs/**/*.zip", recursive=True)))
    col2.metric("Total Backtests", len(glob.glob("logs/backtest/*.png", recursive=True)))
    col3.metric("Status", "Ready")

# ===== DATA EXPLORER PAGE =====
elif page == "Data Explorer":
    st.header("IPO Data Explorer")
    
    data_source = st.radio("Select Data Source", ["Mock Real Data", "Upload CSV"])
    
    try:
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload IPO CSV", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.session_state.csv_data = df
            else:
                st.warning("Please upload a CSV file")
                st.stop()
        else:
            # Use mock data from data_simulator_quant
            from data_simulator_quant import IPODataSimulator
            sim = IPODataSimulator(M=20, feature_dim=4, seed=42)
            features, gains, vols = sim.sample_batch(batch_size=1)
            df = pd.DataFrame({
                "IPO_ID": range(1, 21),
                "Feature_1": features[0][:, 0],
                "Feature_2": features[0][:, 1],
                "Feature_3": features[0][:, 2],
                "Feature_4": features[0][:, 3],
                "Expected_Gain_%": gains[0] * 100,
                "Volatility_20D": vols[0],
            })
        
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total IPOs", len(df))
        col2.metric("Avg Expected Gain", f"{df['Expected_Gain_%'].mean():.2f}%")
        col3.metric("Avg Volatility", f"{df['Volatility_20D'].mean():.4f}")
        
    except Exception as e:
        st.error(f"Error loading data: {e}")

# ===== TRAINING PAGE =====
elif page == "Training":
    st.header("Train PPO Agent")
    st.write("Configure and train a new PPO agent on IPO allocation task.")
    
    with st.form("training_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            timesteps = st.number_input("Training Timesteps", 1000, 1000000, 50000, 1000)
            m_ipos = st.slider("Number of IPOs (M)", 2, 20, 8)
            feature_dim = st.slider("Feature Dimension", 1, 10, 4)
        
        with col2:
            capital = st.number_input("Initial Capital ($)", 10000, 10000000, 1000000, 10000)
            seed = st.number_input("Random Seed", value=123, min_value=1)
            save_dir = st.text_input("Save Directory", "logs/ppo_models")
        
        data_source = st.radio("Data Source", ["Mock Real", "CSV Upload"])
        csv_path = None
        
        if data_source == "CSV Upload":
            uploaded_file = st.file_uploader("Upload IPO CSV", type="csv", key="train_csv")
            if uploaded_file:
                csv_path = uploaded_file.name
                with open(csv_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
        delayed_reward = st.checkbox("Enable Delayed Reward Mode")
        esg = st.checkbox("Enable esg")
        submitted = st.form_submit_button("Start Training")
        
        if submitted:
            st.info("Training started...")
            progress_bar = st.progress(0)
            output_placeholder = st.empty()
            
            cmd = [
                "python", "-u", "train_ppo_quant.py",
                "--timesteps", str(timesteps),
                "--M", str(m_ipos),
                "--capital", str(capital),
                "--feature_dim", str(feature_dim),
                "--seed", str(seed),
                "--save-dir", save_dir,
                "--data_source", "mock_real" if data_source == "Mock Real" else "csv",
            ]
            
            if csv_path:
                cmd.extend(["--csv_path", csv_path])
            
            if delayed_reward:
                cmd.append("--delayed_reward")

            full_output = ""
            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    
                    full_output += line
                    output_placeholder.code(full_output)
                    
                    if "total_timesteps" in line:
                        try:
                            # Extract the number after "total_timesteps |"
                            current_timesteps = int(line.split("total_timesteps")[1].split("|")[1].strip())
                            progress = min(1.0, current_timesteps / timesteps)
                            progress_bar.progress(progress)
                        except (ValueError, IndexError):
                            pass # Ignore parsing errors

                process.wait(timeout=3600)

                if process.returncode == 0:
                    st.success("Training completed successfully!")
                    st.info(f"Model saved to: {save_dir}")
                    progress_bar.progress(1.0)
                else:
                    st.error(f"Training failed. See output above for details.")
                    
            except subprocess.TimeoutExpired:
                st.error("Training timeout (>1 hour)")
            except Exception as e:
                st.error(f"Error: {e}")

# ===== BACKTESTING PAGE =====
elif page == "Backtesting":
    st.header("Backtest Trained Model")
    st.write("Evaluate your trained agent on historical/simulated IPO data.")
    
    # Find available models
    model_files = glob.glob("logs/**/*.zip", recursive=True)
    model_files = [f for f in model_files if "ppo" in os.path.basename(f)]
    
    if not model_files:
        st.warning("No trained models found. Train one first!")
        st.stop()
    
    model_path = st.selectbox("Select Trained Model", model_files)
    
    with st.form("backtest_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            episodes = st.slider("Number of Episodes", 1, 100, 20)
            data_source = st.radio("Data Source", ["Mock Real", "CSV Upload"])
        
        with col2:
            seed = st.number_input("Random Seed", value=42, min_value=1)
            logdir = st.text_input("Log Directory", "logs/backtest")
        
        csv_path = None
        if data_source == "CSV Upload":
            uploaded_file = st.file_uploader("Upload IPO CSV", type="csv", key="backtest_csv")
            if uploaded_file:
                csv_path = uploaded_file.name
                with open(csv_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
        submitted = st.form_submit_button("Start Backtest")
        
        if submitted:
            st.info("Backtesting in progress...")
            
            cmd = [
                "python", "backtest_quant.py",
                "--model", model_path.replace(".zip", ""),
                "--episodes", str(episodes),
                "--seed", str(seed),
                "--logdir", logdir,
                "--data_source", "mock_real" if data_source == "Mock Real" else "csv",
            ]
            
            if csv_path:
                cmd.extend(["--csv_path", csv_path])
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                st.success("Backtest completed!")
                
                # Display backtest output
                with st.expander("Backtest Output"):
                    st.code(result.stdout)
                
                # Look for generated PNG
                png_files = glob.glob(os.path.join(logdir, "*.png"))
                if png_files:
                    latest_png = max(png_files, key=os.path.getctime)
                    st.image(latest_png, caption="Average Wealth Path", use_container_width=True)
                    
                    
            except subprocess.TimeoutExpired:
                st.error("Backtest timeout (>10 min)")
            except Exception as e:
                st.error(f"Error: {e}")

# ===== ALLOCATION ADVISOR PAGE =====
elif page == "Allocation Advisor":
    st.header("AI-Powered IPO Allocation Advisor")
    st.write("Get AI recommendations for IPO allocation using your trained model.")
    
    # Find available models
    model_files = glob.glob("logs/**/*.zip", recursive=True)
    model_files = [f for f in model_files if "ppo" in os.path.basename(f)]
    
    if not model_files:
        st.warning("No trained models found. Train one first!")
        st.stop()
    
    model_path = st.selectbox("Select Model for Allocation", model_files)
    
    # Load model config
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    
    col1, col2 = st.columns(2)
    
    with col1:
        capital = st.number_input("Your Capital ($)", 100000, 10000000, 1000000, 10000)
        data_source = st.radio("IPO Data Source", ["Mock Real", "CSV Upload"], key="advisor_data")
    
    with col2:
        num_ipos = st.slider("Number of IPOs to Consider", 2, 20, config.get("M", 8))
    
    csv_path = None
    if data_source == "CSV Upload":
        uploaded_file = st.file_uploader("Upload IPO CSV", type="csv", key="advisor_csv")
        if uploaded_file:
            csv_path = uploaded_file.name
            with open(csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    
    if st.button("Get AI Allocation Recommendation"):
        st.info("Running model prediction...")
        
        try:
            # Create environment
            def make_env():
                return QuantIPOEnv(
                    M=num_ipos,
                    feature_dim=config.get("feature_dim", 4),
                    initial_capital=capital,
                    real_data_source="csv" if csv_path else "mock_real",
                    csv_path=csv_path,
                    seed=42
                )
            
            env = DummyVecEnv([make_env])
            
            # Load VecNormalize
            vec_file = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
            if os.path.exists(vec_file):
                env = VecNormalize.load(vec_file, env)
                env.training = False
                env.norm_reward = False
            
            # Load model
            model_file = model_path if model_path.endswith(".zip") else model_path + ".zip"
            model = PPO.load(model_file)
            
            # Get prediction
            obs = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            
            # Normalize allocations
            allocation_raw = action[0][:num_ipos]
            allocation_pct = (allocation_raw / (allocation_raw.sum() + 1e-8)) * 100
            amounts = (allocation_pct / 100) * capital
            
            # Display results
            st.success("Allocation Generated!")
            
            result_df = pd.DataFrame({
                "IPO": [f"IPO_{i+1}" for i in range(num_ipos)],
                "Allocation %": allocation_pct,
                "Amount ($)": amounts,
                "Action Value": allocation_raw,
            })
            
            st.dataframe(result_df, use_container_width=True)
            
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            non_zero = allocation_pct[allocation_pct > 0.1]
            if len(non_zero) > 0:
                ax.pie(non_zero, labels=[f"IPO_{i+1}" for i, pct in enumerate(allocation_pct) if pct > 0.1], 
                       autopct='%1.1f%%', startangle=90)
            ax.set_title(f"Recommended IPO Allocation (Total: ${capital:,.0f})")
            st.pyplot(fig)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Allocated", f"${amounts.sum():,.0f}")
            col2.metric("Num IPOs Selected", len(allocation_pct[allocation_pct > 1]))
            col3.metric("Max Allocation", f"{allocation_pct.max():.1f}%")
            
        except Exception as e:
            st.error(f"Error getting allocation: {e}")


