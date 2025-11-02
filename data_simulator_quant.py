import numpy as np
from typing import Tuple, Optional

class IPODataSimulator:
    """
    Simulates batches of IPO opportunities with quant-style features,
    listing returns and allotment fractions.
    """
    def __init__(self, M: int, feature_dim: int = 4, seed: Optional[int] = None):
        """
        M: number of IPOs per time step
        feature_dim: number of features per IPO
        seed: random seed for reproducibility
        """
        if seed is None:
             seed = np.random.randint(0, 1_000_000)
        self.rs = np.random.RandomState(seed)
        self.M = M
        self.feature_dim = feature_dim
        self.rs = np.random.RandomState(seed)
        # Distributions for IPO quantitative data simulation
        # These hyperparameters can be adjusted for realism
        self.feature_means = np.array([0.02, 1.5, 0.5, 0.10])[:feature_dim]  # e.g. moments, popularity, etc.
        self.feature_stds = np.array([0.01, 0.4, 0.3, 0.03])[:feature_dim]
        self.default_vol = 0.2  # Volatility proxy (~20% annualized)
    def seed(self, seed: int):
        """Seed the simulator RNG."""
        self.rs.seed(seed)
    def sample_batch(self, batch_size: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a batch of IPOs (features, expected listing gain, volatility).

        Returns:
            features: (batch_size, M, feature_dim)
            expected_list_gain: (batch_size, M)  # expected profit per $1
            vol_20d: (batch_size, M)  # 20-day volatility per IPO
        """
        features = self.rs.normal(self.feature_means, self.feature_stds, size=(batch_size, self.M, self.feature_dim))
        # Simulate expected listing gain per IPO $1 (mean 8%, stddev 15%)
        expected_list_gain = self.rs.normal(0.08, 0.15, size=(batch_size, self.M))
        vol_20d = np.clip(self.rs.normal(self.default_vol, 0.05, size=(batch_size, self.M)), 0.05, 0.50)
        return features, expected_list_gain, vol_20d
    def sample_realized_listing_gain(self, batch_size: int = 1) -> np.ndarray:
        """
        Simulate actual realized listing return. Could be stochastic around expectation.
        Shape: (batch_size, M)
        """
        # More volatility than expected gain
        actual_gain = self.rs.normal(0.08, 0.22, size=(batch_size, self.M))
        return actual_gain
    def sample_allotment_fraction(self, batch_size: int = 1) -> np.ndarray:
        """
        Simulate random realized allocation fraction per IPO: [0.3, 1.0], beta-like.
        Shape: (batch_size, M)
        """
        # Beta(2, 2) stretched to [0.3, 1.0]; mean ≈ ~0.65
        x = self.rs.beta(2, 2, size=(batch_size, self.M))
        allotment = 0.3 + 0.7 * x
        return allotment
