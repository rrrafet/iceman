"""
Portfolio Configuration Loader

System for loading portfolio configurations from YAML files and building
PortfolioGraph objects using PortfolioBuilderMultiplicative.
"""

import yaml
import os
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class PortfolioConfig:
    """Portfolio configuration from YAML file."""
    name: str
    description: str
    builder_settings: Dict[str, Any]
    data_sources: Dict[str, str]
    data_structure: Dict[str, str]
    components: List[Dict[str, Any]]
    analysis_settings: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'PortfolioConfig':
        """Load portfolio configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            name=data['name'],
            description=data['description'],
            builder_settings=data.get('builder_settings', {}),
            data_sources=data.get('data_sources', {}),
            data_structure=data.get('data_structure', {}),
            components=data.get('components', []),
            analysis_settings=data.get('analysis_settings', {})
        )


def load_portfolio_from_config(config: PortfolioConfig, base_dir: str) -> Dict[str, Any]:
    """
    Build portfolio graph and load data using PortfolioBuilderMultiplicative.
    
    Args:
        config: Portfolio configuration from YAML
        base_dir: Base directory for resolving relative paths
        
    Returns:
        Dictionary containing portfolio_graph, factor_returns, and metadata
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
    
    try:
        from spark.portfolio.builder_multiplicative import PortfolioBuilderMultiplicative
    except ImportError:
        # Fallback path if direct import fails
        sys.path.append('/Users/rafet/Workspace/Spark')
        from spark.portfolio.builder_multiplicative import PortfolioBuilderMultiplicative
    
    # Resolve data source paths
    portfolio_data_path = os.path.join(base_dir, config.data_sources.get('portfolio_data', ''))
    factor_returns_path = os.path.join(base_dir, config.data_sources.get('factor_returns', ''))
    
    # Load portfolio data
    if not os.path.exists(portfolio_data_path):
        raise FileNotFoundError(f"Portfolio data not found: {portfolio_data_path}")
    
    portfolio_df = pd.read_parquet(portfolio_data_path)
    
    # Load factor returns
    if not os.path.exists(factor_returns_path):
        raise FileNotFoundError(f"Factor returns not found: {factor_returns_path}")
    
    factor_returns = pd.read_parquet(factor_returns_path)
    
    # Keep factor returns in long format - pivot will be done when needed for risk analysis
    
    # Apply factor filtering if specified (for long format data)
    factor_subset = config.analysis_settings.get('factor_subset')
    if factor_subset and 'factor_name' in factor_returns.columns:
        factor_returns = factor_returns[factor_returns['factor_name'].isin(factor_subset)]
    
    # Create portfolio builder with settings from config
    builder = PortfolioBuilderMultiplicative(**config.builder_settings)
    
    # Add components from config
    latest_date = portfolio_df[config.data_structure['date_column']].max()
    latest_data = portfolio_df[portfolio_df[config.data_structure['date_column']] == latest_date]
    
    for component_info in config.components:
        path = component_info['path']
        
        # Get weights from latest data
        component_data = latest_data[
            latest_data[config.data_structure['component_id_column']] == path
        ]
        
        portfolio_weight = None
        benchmark_weight = None
        
        if not component_data.empty:
            portfolio_weight = component_data[config.data_structure['portfolio_weight_column']].iloc[0]
            benchmark_weight = component_data[config.data_structure['benchmark_weight_column']].iloc[0]
        
        # Add path to builder
        builder.add_path(
            path=path,
            portfolio_weight=portfolio_weight,
            benchmark_weight=benchmark_weight,
            component_type=component_info.get('component_type', 'auto'),
            is_overlay=component_info.get('is_overlay', False),
            name=component_info.get('name')
        )
    
    # Build the portfolio graph
    portfolio_graph = builder.build()
    
    # Add time series data to the portfolio graph
    _add_time_series_data(portfolio_graph, portfolio_df, config.data_structure)
    
    return {
        'portfolio_graph': portfolio_graph,
        'factor_returns': factor_returns,
        'config': config,
        'portfolio_data': portfolio_df
    }


def _add_time_series_data(portfolio_graph, portfolio_df: pd.DataFrame, data_structure: Dict[str, str]):
    """Add time series data from DataFrame to portfolio graph metrics."""
    try:
        from spark.portfolio.metrics import SeriesMetric
    except ImportError:
        # Handle cases where SeriesMetric might not be available
        return
    
    # Group data by component
    grouped = portfolio_df.groupby(data_structure['component_id_column'])
    
    for component_id, group in grouped:
        if component_id not in portfolio_graph.components:
            continue
            
        # Sort by date
        group = group.sort_values(data_structure['date_column'])
        dates = pd.to_datetime(group[data_structure['date_column']])
        
        # Add portfolio returns
        if data_structure['portfolio_return_column'] in group.columns:
            portfolio_returns = group[data_structure['portfolio_return_column']]
            returns_series = pd.Series(portfolio_returns.values, index=dates, name='portfolio_return')
            portfolio_graph.metric_store.set_metric(
                component_id, 'portfolio_return', SeriesMetric(returns_series)
            )
        
        # Add benchmark returns
        if data_structure['benchmark_return_column'] in group.columns:
            benchmark_returns = group[data_structure['benchmark_return_column']]
            returns_series = pd.Series(benchmark_returns.values, index=dates, name='benchmark_return')
            portfolio_graph.metric_store.set_metric(
                component_id, 'benchmark_return', SeriesMetric(returns_series)
            )


def get_available_configurations(config_dir: str) -> List[Dict[str, str]]:
    """Get list of available portfolio configurations with names and descriptions."""
    portfolio_dir = os.path.join(config_dir, 'portfolios')
    if not os.path.exists(portfolio_dir):
        return []
    
    configs = []
    for filename in os.listdir(portfolio_dir):
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            config_id = filename.replace('.yaml', '').replace('.yml', '')
            try:
                config_path = os.path.join(portfolio_dir, filename)
                config = PortfolioConfig.from_yaml(config_path)
                configs.append({
                    'id': config_id,
                    'name': config.name,
                    'description': config.description
                })
            except Exception:
                # Skip configs that can't be loaded
                configs.append({
                    'id': config_id,
                    'name': config_id.replace('_', ' ').title(),
                    'description': 'Configuration file'
                })
    
    return configs


def load_configuration(config_name: str, config_dir: str) -> PortfolioConfig:
    """Load a specific portfolio configuration by name."""
    portfolio_dir = os.path.join(config_dir, 'portfolios')
    
    # Try both .yaml and .yml extensions
    for ext in ['.yaml', '.yml']:
        config_path = os.path.join(portfolio_dir, f"{config_name}{ext}")
        if os.path.exists(config_path):
            return PortfolioConfig.from_yaml(config_path)
    
    raise FileNotFoundError(f"Portfolio configuration '{config_name}' not found")


def load_portfolio_from_config_name(config_name: str, config_dir: str) -> Dict[str, Any]:
    """
    Load portfolio from configuration name (convenience function).
    
    Args:
        config_name: Name of configuration file (without extension)
        config_dir: Configuration directory path
        
    Returns:
        Dictionary containing portfolio_graph, factor_returns, and metadata
    """
    config = load_configuration(config_name, config_dir)
    # Use config directory as base directory for resolving relative paths
    base_dir = config_dir
    return load_portfolio_from_config(config, base_dir)