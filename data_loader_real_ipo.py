import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import yfinance as yf
from datetime import datetime, timedelta
import gym
import os
import json
import argparse
import streamlit as st
import subprocess
import glob
import sys
from data_loader_real_ipo import RealIPODataLoader

class QuantIPOEnv(gym.Env):
    """RL Environment for allocating capital across REAL IPOs."""
    
    def __init__(self, M: int = 8, feature_dim: int = 4, initial_capital: float = 1_000_000,
                 lambda_risk: float = 1.0, delayed_reward: bool = False, 
                 real_data_source: str = "mock_real", csv_path: Optional[str] = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.M = M
        self.feature_dim = feature_dim
        self.lambda_risk = lambda_risk
        self.initial_capital = initial_capital
        self.capital = float(initial_capital)
        self.delayed_reward = delayed_reward
        
        # Load REAL IPO data instead of simulator
        self.real_loader = RealIPODataLoader(data_source=real_data_source, csv_path=csv_path)
        self.available_ipos = list(range(len(self.real_loader.ipo_data)))
        
        self.seed_value = seed
        if seed is not None:
            np.random.seed(seed)
        
        self.pending = []
        self.time = 0
        
        obs_dim = M * feature_dim + 1
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(M,), dtype=np.float32)
        
        self.ipo_listing_delay = 2
        self.max_steps = 40
        self.ipo_names_in_episode = []
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.seed(seed)
        
        self.capital = float(self.initial_capital)
        self.pending = []
        self.time = 0
        
        # Sample M random IPOs from real data
        ipo_indices = np.random.choice(self.available_ipos, size=self.M, replace=False)
        self.current_ipo_indices = ipo_indices
        self.ipo_names_in_episode = [
            self.real_loader.ipo_data.iloc[i]["Company"] for i in ipo_indices
        ]
        
        features, exp_gain, vol_20d = self.real_loader.get_batch(ipo_indices)
        self.cur_features = features
        self.cur_exp_gain = exp_gain
        self.cur_vol_20d = vol_20d
        
        obs = self._obs()
        return obs, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # ... existing step logic ...
        
        # Sample next M IPOs (real data)
        ipo_indices = np.random.choice(self.available_ipos, size=self.M, replace=False)
        features, exp_gain, vol_20d = self.real_loader.get_batch(ipo_indices)
        self.cur_features = features
        self.cur_exp_gain = exp_gain
        self.cur_vol_20d = vol_20d
        
        # ... rest of existing code ...
        
        return obs, reward, done, truncated, info

class RealIPODataLoader:
    """
    Load real IPO data from multiple sources:
    - NSE/BSE (India) via yfinance
    - Historical IPO listing data
    - Market features (P/E, sector, volume, etc.)
    """
    
    def __init__(self, data_source: str = "csv", csv_path: Optional[str] = None):
        """
        data_source: "csv" (local), "yfinance" (live), or "mock_real" (realistic sample)
        """
        self.data_source = data_source
        self.csv_path = csv_path
        self.ipo_data = None
        self.load_data()
    
    def load_data(self):
        """Load IPO data based on source."""
        if self.data_source == "csv" and self.csv_path:
            self.ipo_data = pd.read_csv(self.csv_path)
        elif self.data_source == "yfinance":
            self.ipo_data = self._fetch_live_ipo_data()
        elif self.data_source == "mock_real":
            self.ipo_data = self._generate_mock_realistic_data()
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")
    
    def _generate_mock_realistic_data(self) -> pd.DataFrame:
        """Generate realistic mock IPO data (for demo/testing)."""
        np.random.seed(42)
        n_ipos = 50
        
        sectors = ["Tech", "Finance", "Healthcare", "Energy", "Retail"]
        companies = [f"Company_{i}" for i in range(1, n_ipos + 1)]
        
        data = {
            "Company": companies,
            "Symbol": [f"SYM{i}" for i in range(n_ipos)],
            "Sector": np.random.choice(sectors, n_ipos),
            "IPO_Date": [
                datetime.now() - timedelta(days=np.random.randint(1, 365))
                for _ in range(n_ipos)
            ],
            "Issue_Price": np.random.uniform(50, 1500, n_ipos),
            "List_Price": np.random.uniform(60, 1800, n_ipos),
            "Listing_Gain_%": np.random.uniform(-10, 80, n_ipos),
            "Market_Cap_Cr": np.random.uniform(100, 10000, n_ipos),
            "P_E_Ratio": np.random.uniform(15, 100, n_ipos),
            "Momentum_Score": np.random.uniform(0.3, 1.0, n_ipos),
            "Subscribe_Times": np.random.uniform(1, 50, n_ipos),
            "20D_Volatility": np.random.uniform(0.05, 0.4, n_ipos),
            "Allotment_Fraction": np.random.uniform(0.3, 1.0, n_ipos),
        }
        return pd.DataFrame(data)
    
    def _fetch_live_ipo_data(self) -> pd.DataFrame:
        """Fetch live IPO data from yfinance (requires active internet)."""
        # Example: Download recent IPO stocks
        symbols = ["NEWTECH.NS", "FINTECH.NS"]  # NSE symbols
        data_list = []
        
        for sym in symbols:
            try:
                df = yf.download(sym, start="2023-01-01", progress=False)
                hist = yf.Ticker(sym).info
                data_list.append({
                    "Symbol": sym,
                    "Company": hist.get("longName", sym),
                    "P_E_Ratio": hist.get("trailingPE", 0),
                    "Market_Cap_Cr": hist.get("marketCap", 0) / 1e7,
                    "20D_Volatility": df["Close"].pct_change().std() * np.sqrt(252),
                })
            except Exception as e:
                print(f"⚠️ Failed to fetch {sym}: {e}")
        
        return pd.DataFrame(data_list)
    
    def get_features_for_ipo(self, ipo_index: int) -> np.ndarray:
        """
        Extract quantitative features for a specific IPO.
        Returns: (feature_dim,) array
        """
        row = self.ipo_data.iloc[ipo_index]
        features = np.array([
            row.get("P_E_Ratio", 0) / 100,  # Normalized P/E
            row.get("Subscribe_Times", 1),   # Subscription multiple
            row.get("Momentum_Score", 0.5),  # Momentum (0-1)
            row.get("Market_Cap_Cr", 0) / 10000,  # Market cap (normalized)
        ], dtype=np.float32)
        return features
    
    def get_expected_return(self, ipo_index: int) -> float:
        """Get expected listing return for IPO."""
        row = self.ipo_data.iloc[ipo_index]
        return row.get("Listing_Gain_%", 0.08) / 100
    
    def get_volatility(self, ipo_index: int) -> float:
        """Get 20-day volatility for IPO."""
        row = self.ipo_data.iloc[ipo_index]
        return row.get("20D_Volatility", 0.2)
    
    def get_batch(self, batch_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get batch of IPO features, expected returns, and volatilities.
        
        Returns:
            features: (len(batch_indices), 4)
            expected_gains: (len(batch_indices),)
            volatilities: (len(batch_indices),)
        """
        features = np.array([self.get_features_for_ipo(i) for i in batch_indices])
        exp_gains = np.array([self.get_expected_return(i) for i in batch_indices])
        vols = np.array([self.get_volatility(i) for i in batch_indices])
        return features, exp_gains, vols
    
    def get_all_ipo_names(self) -> list:
        """Return list of all IPO company names."""
        return self.ipo_data["Company"].tolist()
    
    def get_ipo_by_name(self, name: str) -> Dict:
        """Get IPO data by company name."""
        row = self.ipo_data[self.ipo_data["Company"] == name]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

def main():
    parser = argparse.ArgumentParser(description="Backtest on REAL IPO data")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--feature_dim", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save-dir", type=str, default="logs/ppo_quant_real")
    parser.add_argument("--delayed_reward", type=bool, default=False)
    parser.add_argument("--data_source", type=str, default="mock_real")
    parser.add_argument("--csv_path", type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
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
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # ... existing training code ...
    
    # Save config
    config = {
        "M": args.M, "feature_dim": args.feature_dim, "capital": args.capital,
        "data_source": args.data_source, "csv_path": args.csv_path
    }
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(config, f)

if __name__ == "__main__":
    main()

# Streamlit Dashboard Code
st.set_page_config(layout="wide")
st.title("Quant IPO RL Agent Dashboard - REAL IPO DATA")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Explorer", "Training", "Backtesting", "Allocation Advisor"])

# ===== DATA EXPLORER PAGE =====
if page == "Data Explorer":
    st.header("📊 Real IPO Data Explorer")
    
    data_source = st.radio("Data Source", ["Mock Real Data", "Upload CSV", "Live (yfinance)"])
    
    csv_path = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload IPO CSV", type="csv")
        if uploaded_file:
            csv_path = uploaded_file.name
            with open(csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = RealIPODataLoader(data_source="csv", csv_path=csv_path)
        else:
            st.warning("Please upload a CSV file")
            st.stop()
    else:
        loader = RealIPODataLoader(data_source="mock_real" if data_source == "Mock Real Data" else "yfinance")
    
    df = loader.ipo_data
    st.dataframe(df, use_container_width=True)
    
    st.subheader("📈 Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total IPOs", len(df))
    col2.metric("Avg Listing Gain", f"{df['Listing_Gain_%'].mean():.2f}%")
    col3.metric("Avg Volatility", f"{df['20D_Volatility'].mean():.2f}")
    
    # Filter by sector
    if "Sector" in df.columns:
        sector_filter = st.multiselect("Filter by Sector", df["Sector"].unique())
        if sector_filter:
            df_filtered = df[df["Sector"].isin(sector_filter)]
            st.write(df_filtered)

# ===== TRAINING PAGE =====
elif page == "Training":
    st.header("🏋️ Train PPO Agent on Real IPO Data")
    
    with st.form("training_form"):
        timesteps = st.number_input("Training Timesteps", 1000, 1000000, 100000, 1000)
        m_ipos = st.slider("Number of IPOs (M)", 2, 20, 8)
        capital = st.number_input("Initial Capital ($)", 10000, 10000000, 1000000, 10000)
        feature_dim = st.slider("Feature Dimension", 1, 10, 4)
        seed = st.number_input("Random Seed", value=123)
        save_dir = st.text_input("Save Directory", "logs/ppo_real")
        
        data_source = st.radio("Data Source", ["Mock Real", "CSV Upload", "Live"])
        csv_path = None
        if data_source == "CSV Upload":
            uploaded_file = st.file_uploader("Upload IPO CSV", type="csv", key="train_csv")
            if uploaded_file:
                csv_path = uploaded_file.name
                with open(csv_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        
        delayed_reward = st.checkbox("Enable Delayed Reward")
        submitted = st.form_submit_button("🚀 Start Training")
        
        if submitted:
            cmd = [
                sys.executable, "train_ppo_quant.py",
                # "python", "train_ppo_quant.py",
                "--timesteps", str(timesteps),
                "--M", str(m_ipos),
                "--capital", str(capital),
                "--feature_dim", str(feature_dim),
                "--seed", str(seed),
                "--save-dir", save_dir,
                "--data_source", data_source.lower().replace(" ", "_"),
            ]
            if csv_path:
                cmd.extend(["--csv_path", csv_path])
            if delayed_reward:
                cmd.append("--delayed_reward True")
            
            st.info("Training started... (check output below)")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout_placeholder = st.empty()
            stdout_output = ""
            for line in iter(process.stdout.readline, ''):
                stdout_output += line
                stdout_placeholder.text_area("Output", stdout_output, height=300)
            process.wait()
            st.success("✅ Training completed!")

# ===== ALLOCATION ADVISOR PAGE =====
elif page == "Allocation Advisor":
    st.header("💰 AI-Powered IPO Allocation Advisor")
    
    st.write("Load a trained model and get AI recommendations for IPO allocation.")
    
    model_files = glob.glob("logs/**/*.zip", recursive=True)
    if not model_files:
        st.warning("No trained models. Train one first!")
        st.stop()
    
    model_path = st.selectbox("Select Model", model_files)
    
    data_source = st.radio("IPO Data Source", ["Mock Real", "CSV Upload"])
    csv_path = None
    if data_source == "CSV Upload":
        uploaded_file = st.file_uploader("Upload IPO CSV", type="csv", key="advisor_csv")
        if uploaded_file:
            csv_path = uploaded_file.name
            with open(csv_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    
    loader = RealIPODataLoader(
        data_source="csv" if csv_path else "mock_real",
        csv_path=csv_path
    )
    
    st.subheader("Available IPOs")
    df = loader.ipo_data.head(10)
    st.dataframe(df[["Company", "Listing_Gain_%", "20D_Volatility", "Subscribe_Times"]])
    
    capital_input = st.number_input("Your Investment Capital ($)", 100000, 10000000, 1000000, 10000)
    
    if st.button("🤖 Get AI Allocation Recommendation"):
        st.info("Running model prediction...")
        
        # Run model prediction
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        
        def make_env():
            env = QuantIPOEnv(M=min(8, len(loader.ipo_data)), initial_capital=capital_input,
                            real_data_source="csv" if csv_path else "mock_real", 
                            csv_path=csv_path)
            return env
        
        env = DummyVecEnv([make_env])
        vec_file = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
        if os.path.exists(vec_file):
            env = VecNormalize.load(vec_file, env)
            env.training = False
            env.norm_reward = False
        
        model_file = model_path if model_path.endswith(".zip") else model_path + ".zip"
        model = PPO.load(model_file)
        
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        
        # Display allocation
        st.success("✅ Allocation Generated!")
        
        m_ipos = min(8, len(loader.ipo_data))
        allocation_pct = (action[0][:m_ipos] / action[0][:m_ipos].sum()) * 100
        amounts = (allocation_pct / 100) * capital_input
        
        allocation_df = pd.DataFrame({
            "IPO": loader.ipo_data.iloc[:m_ipos]["Company"].values,
            "Allocation %": allocation_pct,
            "Amount ($)": amounts,
            "Expected Return": loader.ipo_data.iloc[:m_ipos]["Listing_Gain_%"].values,
        })
        
        st.dataframe(allocation_df, use_container_width=True)
        
        # Pie chart
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(allocation_pct, labels=loader.ipo_data.iloc[:m_ipos]["Company"].values, autopct='%1.1f%%')
        ax.set_title("Recommended IPO Allocation")
        st.pyplot(fig)

# ===== BACKTESTING PAGE =====
elif page == "Backtesting":
    # ... existing backtesting code ...
    pass