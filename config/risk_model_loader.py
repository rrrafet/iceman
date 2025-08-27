"""
Risk Model Loader for Maverick UI

Provides interface for loading and managing risk models from the spark-risk
module registry system, with mock data generation for testing.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import yaml

# Add paths for risk module imports
sys.path.append('/Users/rafet/Workspace/Spark')
sys.path.append('/Users/rafet/Workspace/Spark/spark-risk')


class RiskModelLoader:
    """Loader for risk models with mock data generation capabilities."""
    
    def __init__(self, config_dir: str = None):
        """
        Initialize risk model loader.
        
        Args:
            config_dir: Configuration directory for risk model settings
        """
        self.config_dir = config_dir or os.path.dirname(__file__)
        self._available_models = {}
        self._mock_data_cache = {}
        
        # Load available models
        self._discover_models()
    
    def _discover_models(self):
        """Discover available risk models from registry and config."""
        try:
            # Import from spark-risk module
            from spark.risk.definition import list_available_models, get_model
            from spark.risk.model_definitions import model_registry
            
            # Get models from definition module
            definition_models = list_available_models()
            for model_name in definition_models:
                try:
                    model_def = get_model(model_name)
                    self._available_models[model_name] = {
                        'source': 'definition',
                        'name': model_name,
                        'description': f"Pre-defined {model_name} model",
                        'factors': getattr(model_def, 'factor_names', []),
                        'start_date': getattr(model_def, 'start_date', '2020-01-01'),
                        'end_date': getattr(model_def, 'end_date', '2024-12-31'),
                        'frequency': getattr(model_def, 'frequency', 'B'),
                        'model_object': model_def
                    }
                except Exception as e:
                    print(f"Warning: Could not load model {model_name}: {e}")
            
            # Get models from registry
            registry_models = model_registry.list_models()
            for model_name in registry_models:
                if model_name not in self._available_models:  # Don't overwrite definition models
                    try:
                        model_info = model_registry.get_model_info(model_name)
                        self._available_models[model_name] = {
                            'source': 'registry',
                            'name': model_name,
                            'description': model_info.get('description', f'Registry model {model_name}'),
                            'factors': model_info.get('factor_names', []),
                            'start_date': model_info.get('start_date', '2020-01-01'),
                            'end_date': model_info.get('end_date', '2024-12-31'),
                            'frequency': model_info.get('frequency', 'B'),
                            'num_factors': model_info.get('num_factors', 0),
                            'version': model_info.get('version', '1.0')
                        }
                    except Exception as e:
                        print(f"Warning: Could not load registry model {model_name}: {e}")
            
        except ImportError as e:
            print(f"Warning: Could not import spark-risk module: {e}")
            # Fallback to mock models
            self._create_fallback_models()
    
    def _create_fallback_models(self):
        """Create fallback models when spark-risk is not available."""
        # Basic Fama-French model
        self._available_models['fama_french_5'] = {
            'source': 'mock',
            'name': 'Fama-French 5 Factor',
            'description': 'Classic Fama-French 5-factor model (Market, SMB, HML, RMW, CMA)',
            'factors': ['Market', 'SMB', 'HML', 'RMW', 'CMA'],
            'start_date': '2020-01-01',
            'end_date': '2024-12-31',
            'frequency': 'B',
            'num_factors': 5
        }
        
        # Macro factor model
        self._available_models['macro_factors'] = {
            'source': 'mock',
            'name': 'Macro Factor Model',
            'description': 'Macro-economic factor model (Market, Bonds, Dollar, Commodity, Credit)',
            'factors': ['Market', 'Bonds', 'Dollar', 'Commodity', 'Credit'],
            'start_date': '2020-01-01',
            'end_date': '2024-12-31',
            'frequency': 'B',
            'num_factors': 5
        }
        
        # Style factor model
        self._available_models['style_factors'] = {
            'source': 'mock',
            'name': 'Equity Style Factors',
            'description': 'Equity style factor model (Market, Value, Growth, Quality, Momentum, LowVol)',
            'factors': ['Market', 'Value', 'Growth', 'Quality', 'Momentum', 'LowVol'],
            'start_date': '2020-01-01',
            'end_date': '2024-12-31',
            'frequency': 'B',
            'num_factors': 6
        }
        
        # Sector factor model
        self._available_models['sector_model'] = {
            'source': 'mock',
            'name': 'GICS Sector Model',
            'description': 'GICS sector-based factor model with market and 11 sectors',
            'factors': ['Market', 'Energy', 'Materials', 'Industrials', 'ConsumerDiscretionary',
                       'ConsumerStaples', 'HealthCare', 'Financials', 'Technology',
                       'Utilities', 'RealEstate', 'Communication'],
            'start_date': '2020-01-01',
            'end_date': '2024-12-31',
            'frequency': 'B',
            'num_factors': 12
        }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of all available risk models with metadata."""
        return [
            {
                'id': model_id,
                'name': info['name'],
                'description': info['description'],
                'factors': info['factors'],
                'num_factors': len(info['factors']),
                'start_date': info['start_date'],
                'end_date': info['end_date'],
                'frequency': info['frequency'],
                'source': info['source']
            }
            for model_id, info in self._available_models.items()
        ]
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        if model_id not in self._available_models:
            raise KeyError(f"Model '{model_id}' not found. Available: {list(self._available_models.keys())}")
        
        return self._available_models[model_id].copy()
    
    def load_model_factor_returns(self, model_id: str, 
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 frequency: Optional[str] = None) -> pd.DataFrame:
        """
        Load factor returns for a specific model.
        
        Args:
            model_id: Model identifier
            start_date: Override start date (YYYY-MM-DD)
            end_date: Override end date (YYYY-MM-DD) 
            frequency: Override frequency
            
        Returns:
            DataFrame with factor returns indexed by date
        """
        if model_id not in self._available_models:
            raise KeyError(f"Model '{model_id}' not found")
        
        model_info = self._available_models[model_id]
        
        # Use provided parameters or model defaults
        start_date = start_date or model_info['start_date']
        end_date = end_date or model_info['end_date']
        frequency = frequency or model_info['frequency']
        
        # Check cache first
        cache_key = f"{model_id}_{start_date}_{end_date}_{frequency}"
        if cache_key in self._mock_data_cache:
            return self._mock_data_cache[cache_key]
        
        # Try to load from real risk model system
        factor_returns = self._load_real_factor_returns(model_id, start_date, end_date, frequency)
        
        # Fallback to mock data if real system unavailable
        if factor_returns is None:
            factor_returns = self._generate_mock_factor_returns(
                model_info['factors'], start_date, end_date, frequency
            )
        
        # Cache the result
        self._mock_data_cache[cache_key] = factor_returns
        return factor_returns
    
    def _load_real_factor_returns(self, model_id: str, start_date: str, 
                                 end_date: str, frequency: str) -> Optional[pd.DataFrame]:
        """Try to load real factor returns from spark-risk system."""
        try:
            from spark.risk.factor_returns import bloomberg_factor_returns
            
            model_info = self._available_models[model_id]
            
            if model_info['source'] == 'definition':
                # Use model definition object
                return bloomberg_factor_returns(
                    model=model_info['model_object'],
                    start_date=start_date,
                    end_date=end_date,
                    freq=frequency
                )
            elif model_info['source'] == 'registry':
                # Use model name from registry
                return bloomberg_factor_returns(
                    model=model_id,
                    start_date=start_date,
                    end_date=end_date,
                    freq=frequency
                )
                
        except Exception as e:
            print(f"Could not load real factor returns for {model_id}: {e}")
            return None
    
    def _generate_mock_factor_returns(self, factor_names: List[str], 
                                     start_date: str, end_date: str, 
                                     frequency: str) -> pd.DataFrame:
        """Generate realistic mock factor returns."""
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # Generate realistic factor returns
        np.random.seed(42)  # For reproducible results
        
        factor_data = {}
        for i, factor in enumerate(factor_names):
            # Different characteristics for different factor types
            if factor == 'Market':
                # Market factor - higher expected return, higher vol
                returns = np.random.normal(0.0008, 0.012, len(date_range))
            elif factor in ['SMB', 'HML', 'RMW', 'CMA']:
                # Fama-French factors - lower vol, some mean reversion
                returns = np.random.normal(0.0002, 0.006, len(date_range))
            elif factor in ['Bonds', 'Credit']:
                # Fixed income factors - lower vol
                returns = np.random.normal(0.0003, 0.008, len(date_range))
            elif factor in ['Dollar', 'Commodity']:
                # Macro factors - higher vol, some persistence
                returns = np.random.normal(0.0001, 0.015, len(date_range))
            elif factor in ['Energy', 'Materials', 'Utilities']:
                # Commodity-related sectors
                returns = np.random.normal(0.0003, 0.018, len(date_range))
            elif factor == 'Technology':
                # Tech sector - higher growth, higher vol
                returns = np.random.normal(0.0012, 0.020, len(date_range))
            elif factor in ['LowVol', 'Quality']:
                # Defensive factors
                returns = np.random.normal(0.0004, 0.008, len(date_range))
            else:
                # Default factor characteristics
                returns = np.random.normal(0.0005, 0.010, len(date_range))
            
            # Add serial correlation for realism
            for t in range(1, len(returns)):
                returns[t] += 0.1 * returns[t-1] + np.random.normal(0, 0.003)
            
            factor_data[factor] = returns
        
        return pd.DataFrame(factor_data, index=date_range)
    
    def get_model_statistics(self, model_id: str) -> Dict[str, Any]:
        """Get statistical summary of model factor returns."""
        factor_returns = self.load_model_factor_returns(model_id)
        
        # Calculate statistics
        stats = {
            'model_id': model_id,
            'period_start': factor_returns.index.min().strftime('%Y-%m-%d'),
            'period_end': factor_returns.index.max().strftime('%Y-%m-%d'),
            'observations': len(factor_returns),
            'factors': list(factor_returns.columns),
            'factor_stats': {}
        }
        
        for factor in factor_returns.columns:
            series = factor_returns[factor].dropna()
            if len(series) > 0:
                stats['factor_stats'][factor] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'annualized_return': float(series.mean() * 252),
                    'annualized_volatility': float(series.std() * np.sqrt(252))
                }
        
        return stats
    
    def save_model_config(self, model_id: str, config_path: str = None):
        """Save model configuration to YAML file."""
        if config_path is None:
            config_path = os.path.join(self.config_dir, f'risk_model_{model_id}.yaml')
        
        model_info = self.get_model_info(model_id)
        stats = self.get_model_statistics(model_id)
        
        config = {
            'model_id': model_id,
            'name': model_info['name'],
            'description': model_info['description'],
            'source': model_info['source'],
            'parameters': {
                'start_date': model_info['start_date'],
                'end_date': model_info['end_date'],
                'frequency': model_info['frequency']
            },
            'factors': model_info['factors'],
            'statistics': stats['factor_stats']
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        return config_path


def get_default_risk_model_loader() -> RiskModelLoader:
    """Get default risk model loader instance."""
    config_dir = os.path.join(os.path.dirname(__file__))
    return RiskModelLoader(config_dir)


def list_available_risk_models() -> List[Dict[str, Any]]:
    """Convenience function to list available models."""
    loader = get_default_risk_model_loader()
    return loader.get_available_models()