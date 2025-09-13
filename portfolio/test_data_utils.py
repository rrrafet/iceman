"""
Test Data Utilities for Portfolio Testing
========================================

Simple utility functions to generate toy datasets for testing portfolio
graph construction and multiplicative builder functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import PortfolioGraph


def generate_toy_returns(component_ids: List[str], 
                        days: int = 1260, 
                        start_date: str = '2019-01-01') -> Dict[str, Dict[str, pd.Series]]:
    """
    Generate realistic toy returns for portfolio components.
    
    Parameters
    ----------
    component_ids : list of str
        List of component IDs to generate returns for
    days : int, default 1260
        Number of business days to generate (1260 ≈ 5 years)
    start_date : str, default '2019-01-01'
        Start date for the return series
        
    Returns
    -------
    dict
        Dictionary with structure:
        {component_id: {'portfolio_returns': pd.Series, 'benchmark_returns': pd.Series}}
    """
    # Create business day date range
    start = pd.to_datetime(start_date)
    date_range = pd.bdate_range(start=start, periods=days, freq='B')
    
    # Asset class volatility assumptions (annualized)
    volatility_map = {
        'TOTAL': 0.12,
        'EQ': 0.16, 'EQDM': 0.15, 'EQDMLC': 0.14, 'EQDMSC': 0.20,
        'EQEM': 0.22, 'EQSE': 0.16, 'EQSELC': 0.15, 'EQSESC': 0.19,
        'EQLIKE': 0.16,
        'IG': 0.04, 'HY': 0.08,
        'CA': 0.01, 'CCY': 0.06,
        'RE': 0.18, 'RELI': 0.18, 'RENL': 0.20,
        'IN': 0.15, 'INLI': 0.15, 'INNL': 0.18,
        'PE': 0.25, 'HF': 0.12,
        'AS': 0.10, 'OVL': 0.08
    }
    
    # Expected returns (annualized)
    expected_returns_map = {
        'TOTAL': 0.08,
        'EQ': 0.09, 'EQDM': 0.08, 'EQDMLC': 0.08, 'EQDMSC': 0.10,
        'EQEM': 0.11, 'EQSE': 0.08, 'EQSELC': 0.08, 'EQSESC': 0.09,
        'EQLIKE': 0.08,
        'IG': 0.03, 'HY': 0.06,
        'CA': 0.01, 'CCY': 0.02,
        'RE': 0.07, 'RELI': 0.07, 'RENL': 0.08,
        'IN': 0.06, 'INLI': 0.06, 'INNL': 0.07,
        'PE': 0.12, 'HF': 0.05,
        'AS': 0.04, 'OVL': 0.02
    }
    
    # Get correlation matrix for the components
    corr_matrix = generate_realistic_correlations(component_ids)
    
    # Convert annual to daily parameters
    daily_vol = np.array([volatility_map.get(cid, 0.12) for cid in component_ids]) / np.sqrt(252)
    daily_expected = np.array([expected_returns_map.get(cid, 0.06) for cid in component_ids]) / 252
    
    # Create covariance matrix
    cov_matrix = np.outer(daily_vol, daily_vol) * corr_matrix
    
    # Generate portfolio returns
    np.random.seed(42)  # For reproducible results
    portfolio_returns_matrix = np.random.multivariate_normal(
        mean=daily_expected, 
        cov=cov_matrix, 
        size=days
    )
    
    # Generate benchmark returns (similar but with slight differences)
    np.random.seed(43)  # Different seed for benchmark
    benchmark_drift = daily_expected * 0.95  # Slightly lower expected returns
    benchmark_vol = daily_vol * 0.98  # Slightly lower volatility
    benchmark_cov = np.outer(benchmark_vol, benchmark_vol) * corr_matrix
    
    benchmark_returns_matrix = np.random.multivariate_normal(
        mean=benchmark_drift,
        cov=benchmark_cov,
        size=days
    )
    
    # Package into dictionary format
    results = {}
    for i, component_id in enumerate(component_ids):
        portfolio_series = pd.Series(
            portfolio_returns_matrix[:, i], 
            index=date_range, 
            name=f'{component_id}_portfolio_return'
        )
        benchmark_series = pd.Series(
            benchmark_returns_matrix[:, i], 
            index=date_range, 
            name=f'{component_id}_benchmark_return'
        )
        
        results[component_id] = {
            'portfolio_return': portfolio_series,
            'benchmark_return': benchmark_series
        }
    
    return results


def generate_factor_returns(days: int = 1260, 
                           start_date: str = '2019-01-01') -> pd.DataFrame:
    """
    Generate factor model returns for risk decomposition testing.
    
    Parameters
    ----------
    days : int, default 1260
        Number of business days to generate (1260 ≈ 5 years)
    start_date : str, default '2019-01-01'
        Start date for the factor series
        
    Returns
    -------
    pd.DataFrame
        DataFrame with factors as columns and dates as index
    """
    # Create business day date range
    start = pd.to_datetime(start_date)
    date_range = pd.bdate_range(start=start, periods=days, freq='B')
    
    # Factor definitions
    factors = {
        'Market': {'expected': 0.06, 'volatility': 0.15},
        'Size': {'expected': 0.02, 'volatility': 0.08},
        'Value': {'expected': 0.03, 'volatility': 0.10},
        'Momentum': {'expected': 0.04, 'volatility': 0.12},
        'Quality': {'expected': 0.02, 'volatility': 0.06},
        'Low_Vol': {'expected': 0.01, 'volatility': 0.04},
        'US_Region': {'expected': 0.00, 'volatility': 0.05},
        'Europe_Region': {'expected': 0.00, 'volatility': 0.05},
        'EM_Region': {'expected': 0.00, 'volatility': 0.08}
    }
    
    # Factor correlation matrix (simplified)
    factor_names = list(factors.keys())
    n_factors = len(factor_names)
    
    # Create correlation matrix with some realistic correlations
    factor_corr = np.eye(n_factors)
    # Market factor correlates with some others
    factor_corr[0, 1] = -0.3  # Market vs Size (negative)
    factor_corr[1, 0] = -0.3
    factor_corr[0, 2] = -0.1  # Market vs Value (slight negative)
    factor_corr[2, 0] = -0.1
    factor_corr[0, 3] = 0.2   # Market vs Momentum (positive)
    factor_corr[3, 0] = 0.2
    factor_corr[0, 5] = -0.4  # Market vs Low Vol (negative)
    factor_corr[5, 0] = -0.4
    
    # Size and Value correlation
    factor_corr[1, 2] = 0.3
    factor_corr[2, 1] = 0.3
    
    # Regional factors small correlations
    factor_corr[6, 7] = 0.1   # US vs Europe
    factor_corr[7, 6] = 0.1
    factor_corr[6, 8] = 0.05  # US vs EM
    factor_corr[8, 6] = 0.05
    factor_corr[7, 8] = 0.15  # Europe vs EM
    factor_corr[8, 7] = 0.15
    
    # Convert to daily parameters
    daily_expected = np.array([factors[name]['expected'] for name in factor_names]) / 252
    daily_vol = np.array([factors[name]['volatility'] for name in factor_names]) / np.sqrt(252)
    
    # Create covariance matrix
    cov_matrix = np.outer(daily_vol, daily_vol) * factor_corr
    
    # Generate factor returns
    np.random.seed(44)  # Different seed for factors
    factor_returns_matrix = np.random.multivariate_normal(
        mean=daily_expected,
        cov=cov_matrix,
        size=days
    )
    
    # Create DataFrame
    factor_df = pd.DataFrame(
        factor_returns_matrix,
        index=date_range,
        columns=factor_names
    )
    
    return factor_df


def generate_realistic_correlations(component_ids: List[str]) -> np.ndarray:
    """
    Generate realistic correlation matrix for portfolio components.
    
    Parameters
    ----------
    component_ids : list of str
        List of component IDs to create correlations for
        
    Returns
    -------
    np.ndarray
        Correlation matrix (n_components x n_components)
    """
    n = len(component_ids)
    
    # Start with identity matrix
    corr_matrix = np.eye(n)
    
    # Define asset class groupings and their internal correlations
    asset_class_groups = {
        'equity': {'ids': ['EQ', 'EQDM', 'EQDMLC', 'EQDMSC', 'EQEM', 'EQSE', 'EQSELC', 'EQSESC', 'EQLIKE'], 'corr': 0.85},
        'bonds': {'ids': ['IG', 'HY'], 'corr': 0.60},
        'real_estate': {'ids': ['RE', 'RELI', 'RENL'], 'corr': 0.90},
        'infrastructure': {'ids': ['IN', 'INLI', 'INNL'], 'corr': 0.85},
        'alternatives': {'ids': ['PE', 'HF', 'AS'], 'corr': 0.40},
        'cash_currency': {'ids': ['CA', 'CCY'], 'corr': 0.30}
    }
    
    # Cross-asset class correlations
    cross_correlations = {
        ('equity', 'bonds'): 0.15,
        ('equity', 'real_estate'): 0.65,
        ('equity', 'infrastructure'): 0.55,
        ('equity', 'alternatives'): 0.45,
        ('equity', 'cash_currency'): -0.05,
        ('bonds', 'real_estate'): 0.20,
        ('bonds', 'infrastructure'): 0.25,
        ('bonds', 'alternatives'): 0.10,
        ('bonds', 'cash_currency'): 0.05,
        ('real_estate', 'infrastructure'): 0.60,
        ('real_estate', 'alternatives'): 0.35,
        ('real_estate', 'cash_currency'): 0.00,
        ('infrastructure', 'alternatives'): 0.30,
        ('infrastructure', 'cash_currency'): 0.05,
        ('alternatives', 'cash_currency'): 0.00
    }
    
    # Helper function to find asset class for a component
    def get_asset_class(component_id):
        for asset_class, info in asset_class_groups.items():
            if component_id in info['ids']:
                return asset_class
        return 'other'
    
    # Fill correlation matrix
    for i in range(n):
        for j in range(i + 1, n):
            comp_i = component_ids[i]
            comp_j = component_ids[j]
            
            asset_class_i = get_asset_class(comp_i)
            asset_class_j = get_asset_class(comp_j)
            
            if asset_class_i == asset_class_j and asset_class_i != 'other':
                # Same asset class - use internal correlation
                correlation = asset_class_groups[asset_class_i]['corr']
            elif asset_class_i != 'other' and asset_class_j != 'other':
                # Different asset classes - use cross correlation
                key = tuple(sorted([asset_class_i, asset_class_j]))
                correlation = cross_correlations.get(key, 0.20)  # Default cross-correlation
            else:
                # Unknown asset class - use moderate correlation
                correlation = 0.20
            
            # Add some noise to make it more realistic
            np.random.seed(hash(comp_i + comp_j) % 1000)  # Deterministic but varied
            noise = np.random.normal(0, 0.05)
            correlation = np.clip(correlation + noise, -0.95, 0.95)
            
            corr_matrix[i, j] = correlation
            corr_matrix[j, i] = correlation
    
    # Special handling for TOTAL - should be highly correlated with major components
    total_idx = None
    if 'TOTAL' in component_ids:
        total_idx = component_ids.index('TOTAL')
        for i, comp_id in enumerate(component_ids):
            if i != total_idx:
                # TOTAL correlation based on component weight/importance
                if comp_id in ['EQ', 'IG', 'RE']:  # Major components
                    correlation = 0.95
                elif comp_id in ['EQDM', 'EQSE']:  # Large sub-components
                    correlation = 0.90
                else:  # Smaller components
                    correlation = 0.70
                
                corr_matrix[total_idx, i] = correlation
                corr_matrix[i, total_idx] = correlation
    
    # Ensure matrix is positive semidefinite
    eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)  # Ensure all eigenvalues are positive
    corr_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Rescale diagonal to 1
    diag_sqrt = np.sqrt(np.diag(corr_matrix))
    corr_matrix = corr_matrix / np.outer(diag_sqrt, diag_sqrt)
    
    return corr_matrix


def build_test_portfolio(component_ids: List[str], 
                        weights_dict: Dict[str, Dict[str, float]],
                        returns_data: Optional[Dict[str, Dict[str, pd.Series]]] = None,
                        builder_kwargs: Optional[Dict] = None) -> 'PortfolioGraph':
    """
    Quick utility to build a portfolio graph for testing using PortfolioBuilderMultiplicative.
    
    Parameters
    ----------
    component_ids : list of str
        List of component IDs to include in the portfolio
    weights_dict : dict
        Dictionary with component weights: {component_id: {'portfolio_weight': float, 'benchmark_weight': float}}
    returns_data : dict, optional
        Return data from generate_toy_returns(). If None, generates new data.
    builder_kwargs : dict, optional
        Additional arguments to pass to PortfolioBuilderMultiplicative constructor
        
    Returns
    -------
    PortfolioGraph
        Built portfolio graph ready for testing
        
    Examples
    --------
    >>> component_ids = ['TOTAL', 'EQ', 'EQDM', 'EQDMLC']
    >>> weights = {'TOTAL': {'portfolio_weight': 1.0, 'benchmark_weight': 1.0}, ...}
    >>> returns_data = generate_toy_returns(component_ids, days=252)
    >>> graph = build_test_portfolio(component_ids, weights, returns_data)
    """
    # Import here to avoid circular imports
    from .builder_multiplicative import PortfolioBuilderMultiplicative
    
    # Set default builder arguments
    default_builder_kwargs = {
        'normalize_to_relative': True,
        'allow_shorts': True,  # Allow for cash positions
        'root_id': 'TOTAL'
    }
    if builder_kwargs:
        default_builder_kwargs.update(builder_kwargs)
    
    # Create builder
    builder = PortfolioBuilderMultiplicative(**default_builder_kwargs)
    
    # Generate returns data if not provided
    if returns_data is None:
        returns_data = generate_toy_returns(component_ids)
    
    # Infer hierarchy from component relationships
    hierarchy_paths = _infer_hierarchy_paths(component_ids, weights_dict)
    
    # Add components to builder
    for component_id in component_ids:
        # Get weights
        comp_weights = weights_dict.get(component_id, {})
        portfolio_weight = comp_weights.get('portfolio_weight')
        benchmark_weight = comp_weights.get('benchmark_weight')
        
        # Get returns data for this component
        comp_returns = returns_data.get(component_id, {})
        data_dict = {}
        if 'portfolio_return' in comp_returns:
            data_dict['portfolio_return'] = comp_returns['portfolio_return']
        if 'benchmark_return' in comp_returns:
            data_dict['benchmark_return'] = comp_returns['benchmark_return']
        
        # Determine path - use hierarchy if available, otherwise use component_id
        path = hierarchy_paths.get(component_id, component_id)
        
        # Add to builder
        builder.add_path(
            path=path,
            portfolio_weight=portfolio_weight,
            benchmark_weight=benchmark_weight,
            data=data_dict,
            name=component_id
        )
    
    # Build and return the graph
    return builder.build()


def _infer_hierarchy_paths(component_ids: List[str], 
                          _weights_dict: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, str]:
    """
    Infer hierarchical paths from component names and weight structure.
    
    This creates a simple hierarchy based on component naming conventions:
    - TOTAL is root
    - EQ* components are under equity hierarchy
    - Other components are direct children of TOTAL
    
    Note: weights_dict parameter is not used in current implementation but kept for future extensibility
    """
    paths = {}
    
    # Define hierarchy mapping based on your component structure
    hierarchy_map = {
        'TOTAL': 'TOTAL',
        'EQ': 'EQ',
        'EQDM': 'EQ/EQDM',
        'EQDMLC': 'EQ/EQDM/EQDMLC',
        'EQDMSC': 'EQ/EQDM/EQDMSC',
        'EQEM': 'EQ/EQEM',
        'EQSE': 'EQ/EQSE',
        'EQSELC': 'EQ/EQSE/EQSELC',
        'EQSESC': 'EQ/EQSE/EQSESC',
        'EQLIKE': 'EQ/EQLIKE',
        'IG': 'IG',
        'HY': 'HY',
        'AS': 'AS',
        'CA': 'CA',
        'CCY': 'CCY',
        'HF': 'HF',
        'OVL': 'OVL',
        'PE': 'PE',
        'RE': 'RE',
        'RELI': 'RE/RELI',
        'RENL': 'RE/RENL',
        'IN': 'IN',
        'INLI': 'IN/INLI',
        'INNL': 'IN/INNL'
    }
    
    for component_id in component_ids:
        paths[component_id] = hierarchy_map.get(component_id, component_id)
    
    return paths