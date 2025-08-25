import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from services.risk_service import RiskAnalysisService

@dataclass
class SidebarState:
    lens: str = "portfolio"
    selected_node: str = "TOTAL"
    date_range: tuple = (0, 59)  # Default to full range
    selected_factors: List[str] = None
    annualized: bool = True
    show_percentage: bool = True
    
    def __post_init__(self):
        if self.selected_factors is None:
            self.selected_factors = []

class DataLoader:
    def __init__(
        self, 
        data_path: str = None,
        portfolio_graph: object = None,
        factor_returns: object = None,
        use_risk_service: bool = True
    ):
        """
        Initialize DataLoader with multiple data source options.
        
        Args:
            data_path: Path to static data file (fallback)
            portfolio_graph: PortfolioGraph for direct risk analysis
            factor_returns: Factor returns for risk analysis
            use_risk_service: Whether to use RiskAnalysisService for data generation
        """
        self.data_path = data_path
        self.portfolio_graph = portfolio_graph
        self.factor_returns = factor_returns
        self.use_risk_service = use_risk_service
        self._data = None
        self._risk_service = None
        
        # Set default data path if not provided
        if data_path is None:
            self.data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../spark/data.txt"))
        
        # Initialize risk service if components available
        if use_risk_service:
            self._risk_service = RiskAnalysisService(
                portfolio_graph=portfolio_graph,
                factor_returns=factor_returns
            )
        
        self._load_data()
    
    def _load_data(self):
        """
        Load data using the best available method:
        1. RiskAnalysisService (if available)
        2. Static file loading
        3. Mock data fallback
        """
        try:
            # Try RiskAnalysisService first
            if self._risk_service:
                try:
                    self._data = self._risk_service.get_risk_data("TOTAL")
                    print("✓ Data loaded from RiskAnalysisService")
                    return
                except Exception as e:
                    print(f"Warning: RiskAnalysisService failed ({e}). Falling back to static data.")
            
            # Fall back to static file loading
            self._load_static_data()
            
        except Exception as e:
            print(f"Warning: All data loading methods failed ({e}). Using mock data.")
            self._data = self._create_mock_data()
    
    def _load_static_data(self):
        """Load data from static file (original method)."""
        try:
            import sys
            
            # Add the directory to sys.path temporarily
            data_dir = os.path.dirname(self.data_path)
            if data_dir not in sys.path:
                sys.path.insert(0, data_dir)
            
            try:
                # Execute the file content to get the data
                with open(self.data_path, 'r') as f:
                    content = f.read()
                # Try to extract just the dictionary if it's a simple assignment
                if content.strip().startswith('{'):
                    # Use exec with restricted globals for safety
                    local_vars = {}
                    exec(f"data = {content}", {"__builtins__": {}}, local_vars)
                    self._data = local_vars['data']
                else:
                    # Use JSON as fallback
                    self._data = json.loads(content)
                
                print("✓ Data loaded from static file")
                    
            finally:
                # Remove from sys.path
                if data_dir in sys.path:
                    sys.path.remove(data_dir)
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except Exception as e:
            print(f"Error loading static data: {e}")
            raise
    
    @property
    def data(self) -> Dict[str, Any]:
        return self._data
    
    def get_lens_data(self, lens: str) -> Dict[str, Any]:
        """Get data for specified lens (portfolio, benchmark, active)"""
        if lens in self._data:
            return self._data[lens]
        return {}
    
    def get_core_metrics(self, lens: str, component: str = None) -> Dict[str, Any]:
        """Get core risk metrics for lens and component"""
        lens_data = self.get_lens_data(lens)
        if component and component in lens_data.get('core_metrics', {}):
            return lens_data['core_metrics'][component]
        return lens_data.get('core_metrics', {})
    
    def get_factor_names(self) -> List[str]:
        """Get list of available factor names"""
        return self._data.get('identifiers', {}).get('factor_names', [])
    
    def get_component_names(self) -> List[str]:
        """Get list of component names from hierarchy"""
        hierarchy = self._data.get('hierarchy', {})
        component_metadata = hierarchy.get('component_metadata', {})
        return list(component_metadata.keys())
    
    def get_hierarchy_info(self) -> Dict[str, Any]:
        """Get hierarchy structure information"""
        return self._data.get('hierarchy', {})
    
    def get_time_series_data(self, series_type: str, component: str = None) -> List[float]:
        """Get time series data for specified type and component"""
        time_series = self._data.get('time_series', {})
        if series_type in time_series:
            series_data = time_series[series_type]
            if component and component in series_data:
                return series_data[component]
            elif isinstance(series_data, list):
                return series_data
        return []
    
    def get_weights(self, weight_type: str) -> Dict[str, float]:
        """Get weight data (portfolio_weights, benchmark_weights, active_weights)"""
        return self._data.get('weights', {}).get(weight_type, {})
    
    def get_contributions(self, lens: str, contrib_type: str = "by_asset") -> Dict[str, float]:
        """Get contribution data for lens"""
        lens_data = self.get_lens_data(lens)
        return lens_data.get('contributions', {}).get(contrib_type, {})
    
    def get_exposures(self, lens: str) -> Dict[str, float]:
        """Get factor exposures for lens"""
        lens_data = self.get_lens_data(lens)
        return lens_data.get('exposures', {}).get('factor_exposures', {})
    
    def get_matrices(self, matrix_type: str) -> Dict[str, Any]:
        """Get matrix data if available"""
        return self._data.get('matrices', {}).get(matrix_type, {})
    
    def get_correlations(self, corr_type: str) -> Dict[str, Any]:
        """Get correlation data"""
        time_series = self._data.get('time_series', {})
        correlations = time_series.get('correlations', {})
        return correlations.get(corr_type, {})
    
    def get_validation_info(self) -> Dict[str, Any]:
        """Get validation checks and summary"""
        return self._data.get('validation', {})
    
    def filter_data_by_date_range(self, data: List[float], date_range: tuple) -> List[float]:
        """Filter time series data by date range"""
        start_idx, end_idx = date_range
        if isinstance(data, list) and len(data) > end_idx:
            return data[start_idx:end_idx + 1]
        return data
    
    def filter_data_by_factors(self, data: Dict[str, Any], selected_factors: List[str]) -> Dict[str, Any]:
        """Filter data dictionary by selected factors"""
        if not selected_factors:
            return data
        
        filtered_data = {}
        for key, value in data.items():
            if key in selected_factors:
                filtered_data[key] = value
        return filtered_data if filtered_data else data
    
    def refresh_data(self, component: str = "TOTAL") -> Dict[str, Any]:
        """
        Refresh data from RiskAnalysisService, bypassing cache.
        
        Args:
            component: Component to refresh data for
            
        Returns:
            Refreshed data dictionary
        """
        if self._risk_service:
            try:
                self._data = self._risk_service.refresh_data(component)
                print(f"✓ Data refreshed for component: {component}")
                return self._data
            except Exception as e:
                print(f"Error refreshing data: {e}")
        else:
            print("RiskAnalysisService not available for data refresh")
        
        return self._data
    
    def save_schema_to_cache(self, schema_object: object, component: str = "TOTAL") -> str:
        """
        Save a RiskResultSchema object to cache.
        
        Args:
            schema_object: RiskResultSchema instance from notebook
            component: Component ID
            
        Returns:
            Path to cached file
        """
        if self._risk_service:
            return self._risk_service.save_schema_to_cache(schema_object, component)
        else:
            print("RiskAnalysisService not available for caching")
            return ""
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached data.
        
        Returns:
            Cache information dictionary
        """
        if self._risk_service:
            return self._risk_service.get_cache_info()
        else:
            return {"message": "RiskAnalysisService not available"}
    
    def has_risk_service(self) -> bool:
        """Check if RiskAnalysisService is available."""
        return self._risk_service is not None
    
    def set_portfolio_components(self, portfolio_graph: object, factor_returns: object):
        """
        Update portfolio components for dynamic risk analysis.
        
        Args:
            portfolio_graph: New PortfolioGraph instance
            factor_returns: New factor returns data
        """
        self.portfolio_graph = portfolio_graph
        self.factor_returns = factor_returns
        
        # Reinitialize risk service with new components
        if self.use_risk_service:
            self._risk_service = RiskAnalysisService(
                portfolio_graph=portfolio_graph,
                factor_returns=factor_returns
            )
            print("✓ Portfolio components updated, risk service reinitialized")
    
    def _create_mock_data(self) -> Dict[str, Any]:
        """Create mock data for development when real data is not available"""
        return {
            "metadata": {
                "analysis_type": "hierarchical",
                "timestamp": "2025-08-24T20:48:18.867862",
                "data_frequency": "D",
                "annualized": True,
                "schema_version": "2.0"
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
                    "TOTAL": {
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
                    "TOTAL": {
                        "total_risk": 0.0220,
                        "factor_risk_contribution": 0.0160,
                        "specific_risk_contribution": 0.0060,
                        "factor_risk_percentage": 73.0
                    }
                }
            },
            "active": {
                "core_metrics": {
                    "TOTAL": {
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
                    "TOTAL": [0.01, 0.02, -0.01, 0.015, 0.005] * 12  # Mock 60 periods
                },
                "benchmark_returns": {
                    "TOTAL": [0.008, 0.018, -0.008, 0.012, 0.003] * 12
                },
                "active_returns": {
                    "TOTAL": [0.002, 0.002, -0.002, 0.003, 0.002] * 12
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