
import streamlit as st
import subprocess
import os
import glob

st.set_page_config(layout="wide")

st.title("Quant IPO RL Agent Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Training", "Backtesting"])

if page == "Training":
    st.header("Train a New PPO Agent")

    with st.form("training_form"):
        st.write("Configure the training parameters for the PPO agent.")

        timesteps = st.number_input("Number of Training Timesteps", min_value=1000, max_value=1000000, value=100000, step=1000)
        m_ipos = st.slider("Number of IPOs (M)", min_value=2, max_value=20, value=8)
        capital = st.number_input("Initial Capital", min_value=10000, max_value=10000000, value=1000000, step=10000)
        feature_dim = st.slider("IPO Feature Dimension", min_value=1, max_value=10, value=4)
        seed = st.number_input("Random Seed", value=123)
        save_dir = st.text_input("Save Directory", value="logs/ppo_quant_dashboard")
        delayed_reward = st.checkbox("Enable Delayed (Realized) Reward Mode", value=False)

        submitted = st.form_submit_button("Start Training")

        if submitted:
            st.write("Starting training... This may take a while.")
            command = [
                "python", "train_ppo_quant.py",
                "--timesteps", str(timesteps),
                "--M", str(m_ipos),
                "--capital", str(capital),
                "--feature_dim", str(feature_dim),
                "--seed", str(seed),
                "--save-dir", save_dir,
            ]
            if delayed_reward:
                command.append("--delayed_reward True")

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            stdout_placeholder = st.empty()
            stderr_placeholder = st.empty()
            
            stdout_output = ""
            stderr_output = ""

            for line in iter(process.stdout.readline, ''):
                stdout_output += line
                stdout_placeholder.text_area("STDOUT", stdout_output, height=300)

            for line in iter(process.stderr.readline, ''):
                stderr_output += line

            process.wait()

            if process.returncode == 0:
                st.success("Training completed successfully!")
                st.write(f"Model saved in: {save_dir}")
            else:
                st.error("Training failed.")


elif page == "Backtesting":
    st.header("Backtest a Trained Agent")

    # Find available models
    model_files = glob.glob("logs/**/*.zip", recursive=True)
    model_files = [f for f in model_files if "ppo_quant" in os.path.basename(f)]


    if not model_files:
        st.warning("No trained models found. Please train a model first.")
    else:
        with st.form("backtesting_form"):
            st.write("Configure the backtesting parameters.")

            model_path = st.selectbox("Select Model", model_files)
            episodes = st.number_input("Number of Backtest Episodes", min_value=1, max_value=1000, value=50, step=1)
            m_ipos = st.slider("Number of IPOs (M)", min_value=2, max_value=20, value=8)
            capital = st.number_input("Initial Capital", min_value=10000, max_value=10000000, value=1000000, step=10000)
            seed = st.number_input("Random Seed", value=123)
            logdir = st.text_input("Log Directory for Plot", value="logs/backtest_dashboard")
            delayed_reward = st.checkbox("Enable Delayed (Realized) Reward Mode", value=False)

            submitted = st.form_submit_button("Start Backtesting")

            if submitted:
                st.write("Starting backtesting...")
                
                vecnormalize_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
                
                command = [
                    "python", "backtest_quant.py",
                    "--model", model_path,
                    "--episodes", str(episodes),
                    "--M", str(m_ipos),
                    "--capital", str(capital),
                    "--seed", str(seed),
                    "--logdir", logdir,
                ]
                if os.path.exists(vecnormalize_path):
                    command.extend(["--vecnormalize", vecnormalize_path])
                
                if delayed_reward:
                    command.append("--delayed_reward True")

                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                stdout_placeholder = st.empty()
                
                stdout_output = ""
                
                for line in iter(process.stdout.readline, ''):
                    stdout_output += line
                    stdout_placeholder.text_area("STDOUT", stdout_output, height=300)

                process.wait()

                st.success("Backtesting completed successfully!")
                plot_path = os.path.join(logdir, "avg_wealth.png")
                if os.path.exists(plot_path):
                    st.image(plot_path, caption="Average Wealth Path")
                else:
                    st.warning("Plot not found.")
               

