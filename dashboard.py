import streamlit as st
import subprocess
import os
import glob
import time

st.set_page_config(page_title="Quant IPO RL Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stProgress > div > div > div > div {background-color: #007bff;}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Training", "Backtesting"])

if page == "Training":
    st.title("Train PPO Agent for Quant IPO")

    st.info("Configure your PPO agent training parameters below and monitor real-time logs.")

    with st.form("training_form"):
        c1, c2, c3 = st.columns(3)
        timesteps = c1.number_input("Training Timesteps", 1_000, 1_000_000, 100_000, 1_000)
        m_ipos = c2.slider("Number of IPOs (M)", 2, 20, 8)
        feature_dim = c3.slider("IPO Feature Dimension", 1, 10, 4)

        c4, c5, c6 = st.columns(3)
        capital = c4.number_input("Initial Capital", 10_000, 10_000_000, 1_000_000, 10_000)
        seed = c5.number_input("Random Seed", value=123)
        delayed_reward = c6.checkbox("Enable Delayed Reward", value=False)

        save_dir = st.text_input("Model Save Directory", "logs/ppo_quant_dashboard")

        submitted = st.form_submit_button("Start Training")

    if submitted:
        st.subheader("Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        stdout_placeholder = st.empty()

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

        stdout_output = ""
        for i, line in enumerate(iter(process.stdout.readline, '')):
            stdout_output += line
            stdout_placeholder.text_area("Training Logs", stdout_output, height=300)
            progress_bar.progress(min((i % 100) / 100, 1.0))
            status_text.info(f"Processing batch {i}...")

        process.wait()
        progress_bar.progress(1.0)
        status_text.success("Training Complete")

        if process.returncode == 0:
            st.success("Training completed successfully.")
            st.write(f"Model saved at: `{save_dir}`")
        else:
            st.error("Training failed. Check logs above.")

elif page == "Backtesting":
    st.title("Backtest Trained PPO Agent")

    st.info("Select a trained model and run backtesting to evaluate agent performance.")

    # Find trained models
    model_files = glob.glob("logs/**/*.zip", recursive=True)
    model_files = [f for f in model_files if "ppo_quant" in os.path.basename(f)]

    if not model_files:
        st.warning("No trained models found. Train a model first.")
    else:
        with st.form("backtesting_form"):
            c1, c2, c3 = st.columns(3)
            model_path = c1.selectbox("Select Model", model_files)
            episodes = c2.number_input("Backtest Episodes", 1, 1000, 50, 1)
            m_ipos = c3.slider("Number of IPOs (M)", 2, 20, 8)

            c4, c5, c6 = st.columns(3)
            capital = c4.number_input("Initial Capital", 10_000, 10_000_000, 1_000_000, 10_000)
            seed = c5.number_input("Random Seed", value=123)
            delayed_reward = c6.checkbox("Enable Delayed Reward", value=False)

            logdir = st.text_input("Log Directory", "logs/backtest_dashboard")

            submitted = st.form_submit_button("Start Backtesting")

        if submitted:
            st.subheader("Backtesting Progress")
            progress_bar = st.progress(0)
            stdout_placeholder = st.empty()
            stdout_output = ""

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

            for i, line in enumerate(iter(process.stdout.readline, '')):
                stdout_output += line
                stdout_placeholder.text_area("Backtesting Logs", stdout_output, height=300)
                progress_bar.progress(min((i % 100) / 100, 1.0))

            process.wait()
            progress_bar.progress(1.0)
            st.success("Backtesting complete.")

            plot_path = os.path.join(logdir, "avg_wealth.png")
            if os.path.exists(plot_path):
                st.image(plot_path, caption="Average Wealth Over Time", use_container_width=True)
            else:
                st.warning("No plot found in the specified directory.")
