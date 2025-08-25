"""
Risk Analysis Service for Maverick Application

Provides direct integration between Spark risk analysis components
and Streamlit UI, enabling real-time risk data generation.
"""

import sys
import os
import pickle
import joblib
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

# Add Spark modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

try:
    from spark.portfolio.risk_analyzer import PortfolioRiskAnalyzer
    from spark.portfolio.graph import PortfolioGraph
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("Warning: Spark modules not available. Using serialized data only.")

class RiskAnalysisService:
    """
    Service class for generating and managing risk analysis data.
    
    Supports multiple data sources:
    1. Direct analysis from PortfolioGraph + factor returns
    2. Pickled RiskResultSchema objects
    3. JSON serialized data (fallback)
    """
    
    def __init__(
        self,
        portfolio_graph: Optional[object] = None,
        factor_returns: Optional[object] = None,
        cache_dir: str = None
    ):
        """
        Initialize risk analysis service.
        
        Args:
            portfolio_graph: PortfolioGraph instance for direct analysis
            factor_returns: Factor returns data for risk analysis
            cache_dir: Directory for caching serialized results
        """
        self.portfolio_graph = portfolio_graph
        self.factor_returns = factor_returns
        self.analyzer = None
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "../cache")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize analyzer if components available
        if SPARK_AVAILABLE and portfolio_graph is not None:
            self.analyzer = PortfolioRiskAnalyzer(portfolio_graph)
    
    def get_risk_data(self, component: str = "TOTAL", use_cache: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive risk analysis data for specified component.
        
        Args:
            component: Component ID to analyze (default: "TOTAL")
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary containing complete risk analysis data
        """
        cache_key = f"risk_data_{component}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        # Try to load from cache first
        if use_cache and os.path.exists(cache_path):
            try:
                return self._load_from_cache(cache_path)
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_path}: {e}")
        
        # Generate fresh data if analyzer available
        if self.analyzer and self.factor_returns is not None:
            try:
                schema = self.analyzer.get_comprehensive_schema(component, self.factor_returns)
                data = self._serialize_schema_data(schema.data)
                
                # Cache the result
                self._save_to_cache(data, cache_path)
                
                return data
            except Exception as e:
                print(f"Error generating risk data: {e}")
        
        # Fallback to mock data
        print(f"Using mock data for component: {component}")
        return self._create_mock_risk_data(component)
    
    def refresh_data(self, component: str = "TOTAL") -> Dict[str, Any]:
        """
        Force refresh of risk data, bypassing cache.
        
        Args:
            component: Component ID to analyze
            
        Returns:
            Fresh risk analysis data
        """
        return self.get_risk_data(component, use_cache=False)
    
    def save_schema_to_cache(self, schema_object: object, component: str = "TOTAL") -> str:
        """
        Save a RiskResultSchema object to cache for later use.
        
        Args:
            schema_object: RiskResultSchema instance
            component: Component ID for cache naming
            
        Returns:
            Path to saved cache file
        """
        cache_path = os.path.join(self.cache_dir, f"schema_{component}.pkl")
        
        try:
            joblib.dump(schema_object, cache_path)
            print(f"Schema saved to cache: {cache_path}")
            return cache_path
        except Exception as e:
            print(f"Error saving schema to cache: {e}")
            return ""
    
    def load_schema_from_cache(self, component: str = "TOTAL") -> Optional[object]:
        """
        Load RiskResultSchema object from cache.
        
        Args:
            component: Component ID to load
            
        Returns:
            RiskResultSchema object or None if not found
        """
        cache_path = os.path.join(self.cache_dir, f"schema_{component}.pkl")
        
        if os.path.exists(cache_path):
            try:
                return joblib.load(cache_path)
            except Exception as e:
                print(f"Error loading schema from cache: {e}")
        
        return None
    
    def _serialize_schema_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize complex data types (numpy arrays, etc.) for storage/transmission.
        
        Args:
            data: Raw schema data dictionary
            
        Returns:
            Serialized data dictionary
        """
        serialized = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                serialized[key] = self._serialize_schema_data(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = value.item()
            elif isinstance(value, list):
                # Handle lists that might contain numpy objects
                serialized[key] = [
                    item.item() if isinstance(item, (np.integer, np.floating)) 
                    else item.tolist() if isinstance(item, np.ndarray)
                    else item
                    for item in value
                ]
            else:
                serialized[key] = value
        
        return serialized
    
    def _load_from_cache(self, cache_path: str) -> Dict[str, Any]:
        """Load data from pickle cache."""
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def _save_to_cache(self, data: Dict[str, Any], cache_path: str) -> None:
        """Save data to pickle cache."""
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def _create_mock_risk_data(self, component: str) -> Dict[str, Any]:
        """
        Create mock risk data for development/testing.
        
        Args:
            component: Component ID
            
        Returns:
            Mock risk analysis data
        """
        return {
            "metadata": {
                "analysis_type": "hierarchical",
                "timestamp": datetime.now().isoformat(),
                "data_frequency": "D",
                "annualized": True,
                "schema_version": "2.0",
                "component": component
            },
            "identifiers": {
                "factor_names": ["Market", "Size", "Value", "Momentum", "Quality", "Low_Vol"],
                "component_names": ["TOTAL", "CA", "OVL", "IG", "EQLIKE"]
            },
            "hierarchy": {
                "root_component": "TOTAL",
                "component_metadata": {
                    "TOTAL": {"type": "node", "level": 0},
                    "CA": {"type": "node", "level": 1},
                    "OVL": {"type": "leaf", "level": 2},
                    "IG": {"type": "leaf", "level": 2},
                    "EQLIKE": {"type": "leaf", "level": 2}
                },
                "adjacency_list": {
                    "TOTAL": ["CA", "OVL", "IG", "EQLIKE"],
                    "CA": []
                }
            },
            "weights": {
                "portfolio_weights": {"CA": 30.0, "OVL": 25.0, "IG": 25.0, "EQLIKE": 20.0},
                "benchmark_weights": {"CA": 35.0, "OVL": 20.0, "IG": 30.0, "EQLIKE": 15.0},
                "active_weights": {"CA": -5.0, "OVL": 5.0, "IG": -5.0, "EQLIKE": 5.0}
            },
            "portfolio": {
                "core_metrics": {
                    component: {
                        "total_risk": 0.0250,
                        "factor_risk_contribution": 0.0180,
                        "specific_risk_contribution": 0.0070,
                        "factor_risk_percentage": 72.0
                    }
                },
                "contributions": {
                    "by_asset": {"CA": 120.0, "OVL": 80.0, "IG": 60.0, "EQLIKE": 40.0},
                    "by_factor": {"Market": 100.0, "Size": 50.0, "Value": 30.0, "Momentum": 20.0}
                },
                "exposures": {
                    "factor_exposures": {"Market": 0.95, "Size": 0.15, "Value": -0.05, "Momentum": 0.25}
                }
            },
            "benchmark": {
                "core_metrics": {
                    component: {
                        "total_risk": 0.0220,
                        "factor_risk_contribution": 0.0160,
                        "specific_risk_contribution": 0.0060,
                        "factor_risk_percentage": 73.0
                    }
                }
            },
            "active": {
                "core_metrics": {
                    component: {
                        "total_risk": 0.0080,
                        "factor_risk_contribution": 0.0050,
                        "specific_risk_contribution": 0.0030,
                        "factor_risk_percentage": 62.5
                    }
                },
                "contributions": {
                    "by_asset": {"CA": -20.0, "OVL": 15.0, "IG": -10.0, "EQLIKE": 15.0},
                    "by_factor": {"Market": 10.0, "Size": -5.0, "Value": 20.0, "Momentum": -10.0}
                },
                "exposures": {
                    "factor_exposures": {"Market": 0.05, "Size": -0.10, "Value": 0.15, "Momentum": -0.05}
                }
            },
            "time_series": {
                "currency": "USD",
                "metadata": {
                    "start_date": "2023-01-01",
                    "end_date": "2024-12-31"
                },
                "portfolio_returns": {
                    component: [0.01, 0.02, -0.01, 0.015, 0.005] * 12  # Mock 60 periods
                },
                "benchmark_returns": {
                    component: [0.008, 0.018, -0.008, 0.012, 0.003] * 12
                },
                "active_returns": {
                    component: [0.002, 0.002, -0.002, 0.003, 0.002] * 12
                },
                "factor_returns": {
                    "Market": [0.012, 0.015, -0.010, 0.008, 0.005] * 12,
                    "Size": [0.005, -0.002, 0.008, 0.003, -0.001] * 12,
                    "Value": [0.008, 0.010, -0.005, 0.012, 0.007] * 12,
                    "Momentum": [-0.003, 0.005, 0.002, -0.008, 0.004] * 12
                },
                "correlations": {},
                "statistics": {}
            },
            "matrices": {},
            "validation": {
                "checks": {
                    "passes": True,
                    "asset_sum_check": True,
                    "factor_specific_sum_check": True
                },
                "summary": "All validation checks passed"
            }
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data files.
        
        Returns:
            Dictionary with cache file information
        """
        cache_files = []
        
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(self.cache_dir, filename)
                    stat = os.stat(filepath)
                    cache_files.append({
                        'filename': filename,
                        'size_mb': stat.st_size / (1024 * 1024),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'component': filename.replace('risk_data_', '').replace('schema_', '').replace('.pkl', '')
                    })
        
        return {
            'cache_dir': self.cache_dir,
            'cache_files': cache_files,
            'total_files': len(cache_files),
            'spark_available': SPARK_AVAILABLE
        }