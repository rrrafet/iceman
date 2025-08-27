"""
One-off Dummy Data Generator for Portfolio Testing

Generates realistic synthetic portfolio and factor returns data for testing
the YAML-based portfolio configuration system with builder_multiplicative.py
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List

class PortfolioDataGenerator:
    """Generate synthetic portfolio and factor returns data."""
    
    def __init__(self, start_date: str = "2023-01-01", periods: int = 252, seed: int = 42):
        """
        Initialize the data generator.
        
        Parameters
        ----------
        start_date : str
            Start date for time series
        periods : int
            Number of trading days (default 252 = 1 year)
        seed : int
            Random seed for reproducible results
        """
        np.random.seed(seed)
        self.start_date = pd.to_datetime(start_date)
        self.periods = periods
        self.dates = pd.date_range(start=self.start_date, periods=periods, freq='B')  # Business days
        
    def generate_factor_returns(self, model_factors: List[str] = None) -> pd.DataFrame:
        """
        Generate synthetic factor returns.
        
        Parameters
        ----------
        model_factors : List[str], optional
            Specific factors to generate. If None, generates comprehensive set.
        """
        if model_factors:
            factors = model_factors
        else:
            # Default comprehensive factor set
            factors = [
                'Market', 'SMB', 'HML', 'RMW', 'CMA',  # Fama-French 5 factors
                'Momentum', 'Quality', 'LowVol', 'Value', 'Growth',  # Style factors
                'Bonds', 'Dollar', 'Commodity', 'Credit',  # Macro factors
                'Energy', 'Materials', 'Industrials', 'ConsumerDiscretionary',  # Sector factors
                'ConsumerStaples', 'HealthCare', 'Financials', 'Technology',
                'Utilities', 'RealEstate', 'Communication'
            ]
        
        factor_data = {}
        
        # Generate correlated factor returns with realistic properties
        base_correlation = 0.3
        for i, factor in enumerate(factors):
            if factor == 'Market':
                # Market factor - higher volatility, mean-reverting
                returns = np.random.normal(0.0008, 0.012, self.periods)  # ~20% annualized vol
            elif factor in ['SMB', 'HML', 'RMW', 'CMA']:
                # Fama-French factors - lower vol, some correlation to market
                market_influence = 0.2 if factor == 'SMB' else 0.1
                market_returns = factor_data.get('Market', np.zeros(self.periods))
                returns = (np.random.normal(0.0002, 0.006, self.periods) + 
                          market_influence * market_returns)
            elif factor in ['Bonds', 'Credit']:
                # Fixed income factors - lower vol, negative correlation to equity
                returns = np.random.normal(0.0003, 0.008, self.periods)
                if 'Market' in factor_data:
                    returns -= 0.3 * factor_data['Market']  # Negative correlation
            elif factor in ['Dollar', 'Commodity']:
                # Macro factors - higher vol, some autocorrelation
                returns = np.random.normal(0.0001, 0.015, self.periods)
            elif factor in ['Energy', 'Materials', 'Utilities']:
                # Commodity-related sectors - higher vol
                returns = np.random.normal(0.0003, 0.015, self.periods)
            elif factor == 'Technology':
                # Tech - higher growth, higher vol
                returns = np.random.normal(0.0012, 0.018, self.periods)
            elif factor in ['LowVol', 'Quality']:
                # Defensive factors - lower vol
                returns = np.random.normal(0.0004, 0.008, self.periods)
            else:
                # Other factors
                returns = np.random.normal(0.0005, 0.010, self.periods)
            
            # Add some serial correlation for realism
            for t in range(1, len(returns)):
                returns[t] += 0.1 * returns[t-1] + np.random.normal(0, 0.002)
            
            factor_data[factor] = returns
        
        return pd.DataFrame(factor_data, index=self.dates)
    
    def generate_model_specific_data(self, model_factors: List[str], 
                                   output_dir: str, model_name: str = "custom"):
        """Generate factor returns for a specific risk model."""
        print(f"Generating factor returns for {model_name} model...")
        print(f"Factors: {model_factors}")
        
        factor_returns = self.generate_factor_returns(model_factors)
        
        # Save to model-specific file
        os.makedirs(output_dir, exist_ok=True)
        model_file = os.path.join(output_dir, f'factor_returns_{model_name}.parquet')
        factor_returns.to_parquet(model_file)
        
        print(f"Model-specific factor returns saved: {model_file}")
        print(f"Shape: {factor_returns.shape}")
        print(f"Date range: {factor_returns.index.min()} to {factor_returns.index.max()}")
        
        # Display sample statistics
        print(f"\nFactor statistics (annualized):")
        stats = factor_returns.describe()
        for factor in factor_returns.columns:
            mean_annual = factor_returns[factor].mean() * 252
            vol_annual = factor_returns[factor].std() * np.sqrt(252)
            print(f"  {factor:15s}: Mean={mean_annual:6.1%}, Vol={vol_annual:6.1%}")
        
        return factor_returns
    
    def generate_portfolio_data(self) -> pd.DataFrame:
        """Generate synthetic portfolio component data matching YAML structure."""
        
        # Components from strategic_portfolio.yaml structure
        components = [
            'TOTAL',           # Root
            'TOTAL/EQLIKE',    # Equity-like
            'TOTAL/EQLIKE/EQDM',     # Developed Markets Equity
            'TOTAL/EQLIKE/EQDM/EQDMSC',  # Small Cap
            'TOTAL/EQLIKE/EQDM/EQDMLC',  # Large Cap
            'TOTAL/IG',        # Investment Grade
            'TOTAL/CA',        # Cash/Alternatives
            'TOTAL/OVL'        # Overlay
        ]
        
        portfolio_data = []
        
        # Generate realistic weights and returns for each component and date
        for date in self.dates:
            for component_id in components:
                # Determine component level and type
                level = component_id.count('/')
                is_leaf = level >= 2 or component_id in ['TOTAL/IG', 'TOTAL/CA', 'TOTAL/OVL']
                is_overlay = 'OVL' in component_id
                
                # Generate realistic weights based on component type
                if component_id == 'TOTAL':
                    portfolio_weight = 1.0
                    benchmark_weight = 1.0
                elif component_id == 'TOTAL/EQLIKE':
                    portfolio_weight = np.random.uniform(0.55, 0.65)
                    benchmark_weight = np.random.uniform(0.58, 0.62)
                elif component_id == 'TOTAL/EQLIKE/EQDM':
                    portfolio_weight = np.random.uniform(0.45, 0.55)
                    benchmark_weight = np.random.uniform(0.48, 0.52)
                elif component_id == 'TOTAL/EQLIKE/EQDM/EQDMSC':
                    portfolio_weight = np.random.uniform(0.15, 0.25)
                    benchmark_weight = np.random.uniform(0.18, 0.22)
                elif component_id == 'TOTAL/EQLIKE/EQDM/EQDMLC':
                    portfolio_weight = np.random.uniform(0.25, 0.35)
                    benchmark_weight = np.random.uniform(0.28, 0.32)
                elif component_id == 'TOTAL/IG':
                    portfolio_weight = np.random.uniform(0.25, 0.35)
                    benchmark_weight = np.random.uniform(0.28, 0.32)
                elif component_id == 'TOTAL/CA':
                    portfolio_weight = np.random.uniform(0.08, 0.12)
                    benchmark_weight = np.random.uniform(0.05, 0.08)
                elif is_overlay:
                    portfolio_weight = np.random.uniform(-0.02, 0.02)  # Small overlay positions
                    benchmark_weight = 0.0
                else:
                    portfolio_weight = np.random.uniform(0.05, 0.15)
                    benchmark_weight = np.random.uniform(0.08, 0.12)
                
                # Generate realistic returns based on component type
                if 'EQLIKE' in component_id or 'EQDM' in component_id:
                    # Equity-like components - higher expected returns and volatility
                    base_return = np.random.normal(0.0008, 0.015)
                    if 'SC' in component_id:  # Small cap
                        base_return += np.random.normal(0.0002, 0.005)  # Small cap premium
                elif 'IG' in component_id:
                    # Investment grade bonds - lower returns, lower vol
                    base_return = np.random.normal(0.0003, 0.008)
                elif 'CA' in component_id:
                    # Cash/alternatives - very low vol
                    base_return = np.random.normal(0.0001, 0.002)
                elif is_overlay:
                    # Overlay - can be volatile
                    base_return = np.random.normal(0.0000, 0.012)
                else:
                    # Default
                    base_return = np.random.normal(0.0005, 0.010)
                
                portfolio_return = base_return
                benchmark_return = base_return + np.random.normal(0, 0.003)  # Small difference
                
                portfolio_data.append({
                    'component_id': component_id,
                    'date': date,
                    'portfolio_weight': portfolio_weight,
                    'benchmark_weight': benchmark_weight,
                    'portfolio_return': portfolio_return,
                    'benchmark_return': benchmark_return
                })
        
        return pd.DataFrame(portfolio_data)
    
    def generate_all_data(self, output_dir: str):
        """Generate all data files and save to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating factor returns data...")
        factor_returns = self.generate_factor_returns()
        factor_returns_path = os.path.join(output_dir, 'factor_returns.parquet')
        factor_returns.to_parquet(factor_returns_path)
        print(f"Factor returns saved to: {factor_returns_path}")
        print(f"Factor returns shape: {factor_returns.shape}")
        
        print("\nGenerating portfolio data...")
        portfolio_data = self.generate_portfolio_data()
        portfolio_data_path = os.path.join(output_dir, 'portfolio.parquet')
        portfolio_data.to_parquet(portfolio_data_path, index=False)
        print(f"Portfolio data saved to: {portfolio_data_path}")
        print(f"Portfolio data shape: {portfolio_data.shape}")
        
        # Display sample data
        print(f"\nSample factor returns (first 5 rows, first 5 columns):")
        print(factor_returns.iloc[:5, :5])
        
        print(f"\nSample portfolio data (first 10 rows):")
        print(portfolio_data.head(10))
        
        print(f"\nComponent weight ranges:")
        weight_summary = portfolio_data.groupby('component_id').agg({
            'portfolio_weight': ['min', 'max', 'mean'],
            'benchmark_weight': ['min', 'max', 'mean']
        }).round(4)
        print(weight_summary)


def main():
    """Main function to generate all data files."""
    print("Portfolio Data Generator")
    print("=" * 50)
    
    # Get the data directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # Generate data
    generator = PortfolioDataGenerator(
        start_date="2023-01-01",
        periods=252,  # 1 year of trading days
        seed=42
    )
    
    # Generate general portfolio data
    generator.generate_all_data(data_dir)
    
    print("\n" + "=" * 40)
    print("Generating risk model specific data...")
    
    # Generate model-specific factor returns
    risk_models = {
        'fama_french_5': ['Market', 'SMB', 'HML', 'RMW', 'CMA'],
        'macro_factors': ['Market', 'Bonds', 'Dollar', 'Commodity', 'Credit'],
        'style_factors': ['Market', 'Value', 'Growth', 'Quality', 'Momentum', 'LowVol'],
        'sector_model': ['Market', 'Energy', 'Materials', 'Industrials', 'ConsumerDiscretionary',
                        'ConsumerStaples', 'HealthCare', 'Financials', 'Technology',
                        'Utilities', 'RealEstate', 'Communication']
    }
    
    for model_name, factors in risk_models.items():
        generator.generate_model_specific_data(factors, data_dir, model_name)
        print()  # Add spacing
    
    print("=" * 50)
    print("Data generation complete!")
    print(f"Files created in: {data_dir}")
    print("- factor_returns.parquet (comprehensive)")
    print("- portfolio.parquet")
    for model_name in risk_models.keys():
        print(f"- factor_returns_{model_name}.parquet")


if __name__ == "__main__":
    main()