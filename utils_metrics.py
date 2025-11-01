import numpy as np
from typing import Optional

def annualized_return(wealth: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return from wealth series.
    Args:
        wealth: array of total wealth at each time step
        periods_per_year: trading periods per year (default 252)
    Returns:
        Annualized return as a float
    """
    if len(wealth) < 2 or wealth[0] <= 0:
        return 0.0
    num_years = len(wealth) / periods_per_year
    if num_years == 0:
        return 0.0
    ratio = wealth[-1] / wealth[0]
    # If ratio is negative, result of exponentiation is NaN. Cap at 0.
    # This effectively caps the total return at -100%.
    if ratio < 0:
        ratio = 0
    return ratio ** (1 / num_years) - 1.0

def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Compute annualized Sharpe ratio for a return series.
	Args:
		returns: Periodic returns (e.g. daily)
		risk_free: Risk free rate per period
	Returns:
		Sharpe ratio (annualized)
    """
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free
    mean = np.mean(excess)
    std = np.std(excess)
    if std == 0:
        return 0.0
    return mean / std * np.sqrt(periods_per_year)

def max_drawdown(wealth: np.ndarray) -> float:
    """
    Calculate maximum drawdown of a wealth time series.
	Args:
		wealth: Sequence of wealth values
	Returns:
		Max drawdown (as a positive float, e.g. 0.3 = 30%)
    """
    if len(wealth) < 2:
        return 0.0
    peak = np.maximum.accumulate(wealth)
    drawdown = (peak - wealth) / np.maximum(peak, 1e-10)
    return np.max(drawdown)
