"""
Script to create mock data files for portfolio risk analysis system testing.
Creates factor_returns.parquet and portfolio.parquet with realistic synthetic data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_factor_returns(start_date: str = "2020-01-01", end_date: str = "2023-12-31") -> pd.DataFrame:
    """
    Create mock factor returns data in long format.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with factor returns data
    """
    # Create date range (business days only)
    dates = pd.bdate_range(start=start_date, end=end_date)
    
    # Define factors for macro1 risk model
    macro1_factors = [
        "EQUITY_MOMENTUM", "EQUITY_VALUE", "EQUITY_QUALITY", "EQUITY_SIZE",
        "CREDIT_SPREAD", "INTEREST_RATE", "CURRENCY", "COMMODITY",
        "VOLATILITY", "LIQUIDITY", "MARKET_NEUTRAL", "SECTOR_ROTATION"
    ]
    
    # Factor characteristics (annual volatility and mean return)
    factor_params = {
        "EQUITY_MOMENTUM": {"vol": 0.12, "mean": 0.03},
        "EQUITY_VALUE": {"vol": 0.15, "mean": 0.05},
        "EQUITY_QUALITY": {"vol": 0.10, "mean": 0.02},
        "EQUITY_SIZE": {"vol": 0.18, "mean": 0.01},
        "CREDIT_SPREAD": {"vol": 0.08, "mean": 0.02},
        "INTEREST_RATE": {"vol": 0.06, "mean": 0.00},
        "CURRENCY": {"vol": 0.09, "mean": 0.00},
        "COMMODITY": {"vol": 0.22, "mean": 0.04},
        "VOLATILITY": {"vol": 0.25, "mean": -0.01},
        "LIQUIDITY": {"vol": 0.14, "mean": 0.01},
        "MARKET_NEUTRAL": {"vol": 0.05, "mean": 0.00},
        "SECTOR_ROTATION": {"vol": 0.11, "mean": 0.02}
    }
    
    # Generate correlated factor returns
    np.random.seed(42)  # For reproducible results
    
    # Create correlation matrix (some factors should be correlated)
    n_factors = len(macro1_factors)
    correlation_matrix = np.eye(n_factors)
    
    # Add some realistic correlations
    correlation_matrix[0, 1] = -0.3  # Momentum vs Value
    correlation_matrix[1, 0] = -0.3
    correlation_matrix[2, 3] = 0.2   # Quality vs Size
    correlation_matrix[3, 2] = 0.2
    correlation_matrix[4, 5] = 0.4   # Credit vs Interest Rate
    correlation_matrix[5, 4] = 0.4
    
    # Generate multivariate normal returns
    daily_returns_matrix = np.random.multivariate_normal(
        mean=[0] * n_factors,
        cov=correlation_matrix,
        size=len(dates)
    )
    
    # Scale by volatility and add drift
    for i, factor in enumerate(macro1_factors):
        params = factor_params[factor]
        daily_vol = params["vol"] / np.sqrt(252)
        daily_mean = params["mean"] / 252
        
        daily_returns_matrix[:, i] = daily_returns_matrix[:, i] * daily_vol + daily_mean
    
    # Convert to long format DataFrame
    records = []
    for i, date in enumerate(dates):
        for j, factor in enumerate(macro1_factors):
            records.append({
                "date": date,
                "factor_name": factor,
                "return_value": daily_returns_matrix[i, j],
                "riskmodel_code": "macro1"
            })
    
    factor_returns_df = pd.DataFrame(records)
    
    logger.info(f"Created factor returns data: {len(factor_returns_df)} records, "
                f"{len(dates)} dates, {len(macro1_factors)} factors")
    
    return factor_returns_df


def create_mock_portfolio_data(start_date: str = "2020-01-01", end_date: str = "2023-12-31") -> pd.DataFrame:
    """
    Create mock portfolio data with hierarchical components.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame with portfolio data
    """
    # Create date range (business days only)
    dates = pd.bdate_range(start=start_date, end=end_date)
    
    # Define portfolio components (hierarchical structure matching strategic_portfolio.yaml)
    components = [
        {"component_id": "TOTAL", "type": "node", "children": ["EQLIKE", "IG", "CA", "OVL"]},
        {"component_id": "EQLIKE", "type": "node", "children": ["EQDM"], "parent": "TOTAL"},
        {"component_id": "EQDM", "type": "node", "children": ["EQDMSC", "EQDMLC"], "parent": "EQLIKE"},
        {"component_id": "EQDMSC", "type": "leaf", "parent": "EQDM"},
        {"component_id": "EQDMLC", "type": "leaf", "parent": "EQDM"},
        {"component_id": "IG", "type": "leaf", "parent": "TOTAL"},
        {"component_id": "CA", "type": "leaf", "parent": "TOTAL"},
        {"component_id": "OVL", "type": "leaf", "parent": "TOTAL"}
    ]
    
    # Define static weights (will vary slightly over time)
    base_weights = {
        "TOTAL": {"portfolio": 1.0, "benchmark": 1.0},
        "EQLIKE": {"portfolio": 0.65, "benchmark": 0.60},
        "EQDM": {"portfolio": 0.65, "benchmark": 0.60},  # Same as EQLIKE for simplicity
        "EQDMSC": {"portfolio": 0.15, "benchmark": 0.10},
        "EQDMLC": {"portfolio": 0.50, "benchmark": 0.50},
        "IG": {"portfolio": 0.25, "benchmark": 0.30},
        "CA": {"portfolio": 0.05, "benchmark": 0.08},
        "OVL": {"portfolio": 0.05, "benchmark": 0.02}
    }
    
    # Define return characteristics for each component
    return_params = {
        "TOTAL": {"vol": 0.12, "mean": 0.06},
        "EQLIKE": {"vol": 0.16, "mean": 0.08},
        "EQDM": {"vol": 0.16, "mean": 0.08},
        "EQDMSC": {"vol": 0.22, "mean": 0.09},
        "EQDMLC": {"vol": 0.15, "mean": 0.07},
        "IG": {"vol": 0.04, "mean": 0.03},
        "CA": {"vol": 0.02, "mean": 0.01},
        "OVL": {"vol": 0.08, "mean": 0.02}
    }
    
    np.random.seed(42)  # For reproducible results
    
    records = []
    
    for component in components:
        component_id = component["component_id"]
        params = return_params[component_id]
        base_weight = base_weights[component_id]
        
        # Generate returns with some autocorrelation
        daily_vol = params["vol"] / np.sqrt(252)
        daily_mean = params["mean"] / 252
        
        # Generate portfolio returns
        portfolio_returns = np.random.normal(daily_mean, daily_vol, len(dates))
        
        # Add some autocorrelation
        for i in range(1, len(portfolio_returns)):
            portfolio_returns[i] += 0.05 * portfolio_returns[i-1]
        
        # Generate benchmark returns (slightly different)
        benchmark_returns = portfolio_returns + np.random.normal(0, daily_vol * 0.3, len(dates))
        
        # Generate time-varying weights (small variations around base weight)
        portfolio_weight_series = base_weight["portfolio"] + np.random.normal(0, 0.01, len(dates))
        benchmark_weight_series = base_weight["benchmark"] + np.random.normal(0, 0.005, len(dates))
        
        # Ensure weights stay reasonable
        portfolio_weight_series = np.clip(portfolio_weight_series, 0, 1)
        benchmark_weight_series = np.clip(benchmark_weight_series, 0, 1)
        
        # Create records for this component
        for i, date in enumerate(dates):
            records.append({
                "component_id": component_id,
                "date": date,
                "portfolio_return": portfolio_returns[i],
                "benchmark_return": benchmark_returns[i],
                "portfolio_weight": portfolio_weight_series[i],
                "benchmark_weight": benchmark_weight_series[i]
            })
    
    portfolio_df = pd.DataFrame(records)
    
    logger.info(f"Created portfolio data: {len(portfolio_df)} records, "
                f"{len(dates)} dates, {len(components)} components")
    
    return portfolio_df


def save_mock_data(data_dir: str = "data") -> None:
    """
    Create and save mock data files.
    
    Args:
        data_dir: Directory to save data files
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    logger.info("Creating mock factor returns data...")
    factor_returns = create_mock_factor_returns()
    
    factor_returns_path = data_path / "factor_returns.parquet"
    factor_returns.to_parquet(factor_returns_path, index=False)
    logger.info(f"Saved factor returns to {factor_returns_path}")
    
    logger.info("Creating mock portfolio data...")
    portfolio_data = create_mock_portfolio_data()
    
    portfolio_data_path = data_path / "portfolio.parquet"
    portfolio_data.to_parquet(portfolio_data_path, index=False)
    logger.info(f"Saved portfolio data to {portfolio_data_path}")
    
    # Print data summaries
    print("\n=== FACTOR RETURNS DATA SUMMARY ===")
    print(f"Shape: {factor_returns.shape}")
    print(f"Date range: {factor_returns['date'].min()} to {factor_returns['date'].max()}")
    print(f"Risk models: {factor_returns['riskmodel_code'].unique()}")
    print(f"Factors: {sorted(factor_returns['factor_name'].unique())}")
    print("\nSample data:")
    print(factor_returns.head())
    
    print("\n=== PORTFOLIO DATA SUMMARY ===")
    print(f"Shape: {portfolio_data.shape}")
    print(f"Date range: {portfolio_data['date'].min()} to {portfolio_data['date'].max()}")
    print(f"Components: {sorted(portfolio_data['component_id'].unique())}")
    print("\nSample data:")
    print(portfolio_data.head())
    
    print("\n=== DATA VALIDATION ===")
    # Check for missing values
    print(f"Factor returns missing values: {factor_returns.isnull().sum().sum()}")
    print(f"Portfolio data missing values: {portfolio_data.isnull().sum().sum()}")
    
    # Check data types
    print(f"\nFactor returns dtypes:\n{factor_returns.dtypes}")
    print(f"\nPortfolio data dtypes:\n{portfolio_data.dtypes}")


if __name__ == "__main__":
    import os
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    save_mock_data()
    print("\nMock data creation completed successfully!")