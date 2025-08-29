"""
Minimal Schema-Only Data Loader for Maverick UI

This module provides a radically simplified interface that uses only the
RiskResultSchema from the schema factory as the single source of truth.
All complex legacy data transformation and mapping logic has been removed.
"""

import os
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class SidebarState:
    """State object containing sidebar filter selections."""
    lens: str
    selected_node: str
    date_range: tuple
    selected_factors: List[str]
    selected_risk_model: str
    annualized: bool
    show_percentage: bool


class DataLoader:
    """Minimal data loader using RiskResultSchema as single source of truth."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.portfolio_graph = None
        self.config = None
        self._current_config_id = None
        self.risk_service = None
        self._risk_schema = None  # Single source of truth
        self._time_series_cache = None  # Cached time series DataFrame
        self._factor_returns = None  # Long-format factor returns DataFrame
        self._current_risk_model = None  # Currently loaded risk model ID
        
        # Try to load default configuration
        self._load_default_configuration()
    
    def _load_default_configuration(self):
        """Load default configuration on initialization."""
        try:
            from config.portfolio_loader import get_available_configurations, load_portfolio_from_config_name
            from services.risk_service import create_risk_service
            
            config_dir = os.path.join(os.path.dirname(__file__), 'config')
            available_configs = get_available_configurations(config_dir)
            
            if available_configs:
                default_config = available_configs[0]['id']
                result = load_portfolio_from_config_name(default_config, config_dir)
                
                self.portfolio_graph = result['portfolio_graph']
                self._factor_returns = result['factor_returns']  # Store long-format factor returns
                self.config = result['config']
                self._current_config_id = default_config
                
                # Initialize risk service
                self.risk_service = create_risk_service()
                if self.risk_service:
                    self.risk_service.set_portfolio_graph(self.portfolio_graph)
                    # Set default risk model (first available) and run analysis
                    available_models = self.get_available_risk_models()
                    if available_models:
                        self._current_risk_model = available_models[0]['id']
                        self._setup_risk_service_and_run_analysis(force_refresh=False)
                    else:
                        print("No risk models available - cannot initialize risk service")
                
        except Exception as e:
            print(f"Could not load default configuration: {e}")
    
    def load_portfolio_configuration(self, config_id: str) -> bool:
        """Load portfolio configuration and run risk analysis."""
        try:
            from config.portfolio_loader import load_portfolio_from_config_name
            
            config_dir = os.path.join(os.path.dirname(__file__), 'config')
            result = load_portfolio_from_config_name(config_id, config_dir)
            
            self.portfolio_graph = result['portfolio_graph']
            self._factor_returns = result['factor_returns']  # Store long-format factor returns
            self.config = result['config']
            self._current_config_id = config_id
            
            # Update risk service and get fresh schema
            if self.risk_service:
                self.risk_service.set_portfolio_graph(self.portfolio_graph)
                # Keep current risk model or set first available
                available_models = self.get_available_risk_models()
                if available_models:
                    if not self._current_risk_model or self._current_risk_model not in [m['id'] for m in available_models]:
                        self._current_risk_model = available_models[0]['id']
                    self._setup_risk_service_and_run_analysis(force_refresh=True)
            
            return True
            
        except Exception as e:
            st.error(f"Failed to load configuration '{config_id}': {str(e)}")
            return False
    
    # =====================================================================
    # MINIMAL SCHEMA-ONLY INTERFACE - 3 Core Methods Only
    # =====================================================================
    
    def get_schema_data(self, component_id: str, lens: str) -> Dict[str, Any]:
        """
        Get schema data for a specific component and lens.
        
        This is the primary data access method that all UI components should use.
        """
        if not self._risk_schema:
            return {}
        
        return self._risk_schema.get_component_lens_data(component_id, lens) or {}
    
    def get_all_components(self) -> List[str]:
        """Get list of all component IDs."""
        if not self._risk_schema:
            return [self.root_component_id]
        
        return self._risk_schema.get_all_components()
    
    def get_factor_names(self) -> List[str]:
        """Get factor names from multiple sources in priority order."""
        # Priority 1: Schema factor names (if populated)
        if self._risk_schema and self._risk_schema.factor_names:
            return self._risk_schema.factor_names
        
        # Priority 2: Extract from factor returns DataFrame
        elif self._factor_returns is not None and 'factor_name' in self._factor_returns.columns:
            return self._factor_returns['factor_name'].unique().tolist()
        
        # Priority 3: Extract from factor exposures in schema data
        elif self._risk_schema:
            components = self._risk_schema.get_all_components()
            for comp_id in components:
                portfolio_data = self._risk_schema.get_component_lens_data(comp_id, 'portfolio')
                if portfolio_data and 'factor_exposures' in portfolio_data:
                    factor_exposures = portfolio_data['factor_exposures']
                    if factor_exposures:
                        return list(factor_exposures.keys())
        
        # Fallback
        return ['Market', 'Bonds', 'Dollar', 'Commodity', 'Credit']
    
    def _build_time_series_cache(self) -> pd.DataFrame:
        """
        Build and cache all portfolio and benchmark return time series from hierarchical schema.
        
        Returns
        -------
        pd.DataFrame
            Long-form DataFrame with columns: component_id, date, lens, return_value
            where lens is 'portfolio', 'benchmark', or 'active'
        """
        if not self._risk_schema:
            return pd.DataFrame(columns=['component_id', 'date', 'lens', 'return_value'])
        
        all_data = []
        
        for component_id in self._risk_schema.get_all_components():
            # Get portfolio time series
            portfolio_data = self.get_schema_data(component_id, 'portfolio')
            portfolio_returns = portfolio_data.get('portfolio_return', [])
            portfolio_dates = portfolio_data.get('portfolio_return_dates', [])
            
            # Get benchmark time series
            benchmark_data = self.get_schema_data(component_id, 'benchmark')
            benchmark_returns = benchmark_data.get('benchmark_return', [])
            benchmark_dates = benchmark_data.get('benchmark_return_dates', [])
            
            # Create portfolio series if data exists
            portfolio_series = None
            if portfolio_returns and portfolio_dates and len(portfolio_returns) == len(portfolio_dates):
                portfolio_series = pd.Series(index=portfolio_dates, data=portfolio_returns)
                for date, return_value in portfolio_series.items():
                    all_data.append({
                        'component_id': component_id,
                        'date': date,
                        'lens': 'portfolio',
                        'return_value': return_value
                    })
            
            # Create benchmark series if data exists
            benchmark_series = None
            if benchmark_returns and benchmark_dates and len(benchmark_returns) == len(benchmark_dates):
                benchmark_series = pd.Series(index=benchmark_dates, data=benchmark_returns)
                for date, return_value in benchmark_series.items():
                    all_data.append({
                        'component_id': component_id,
                        'date': date,
                        'lens': 'benchmark',
                        'return_value': return_value
                    })
            
            # Calculate active returns if both series exist
            if portfolio_series is not None and benchmark_series is not None:
                # Align indices and calculate difference
                active_series = portfolio_series - benchmark_series
                for date, return_value in active_series.items():
                    all_data.append({
                        'component_id': component_id,
                        'date': date,
                        'lens': 'active',
                        'return_value': return_value
                    })
        
        return pd.DataFrame(all_data)
    
    def get_time_series_for_components(self, component_ids: List[str]) -> pd.DataFrame:
        """
        Get time series data for specific components.
        
        Parameters
        ----------
        component_ids : List[str]
            List of component IDs to filter for
            
        Returns
        -------
        pd.DataFrame
            Filtered time series DataFrame for the specified components
        """
        if self._time_series_cache is None or self._time_series_cache.empty:
            return pd.DataFrame(columns=['component_id', 'date', 'lens', 'return_value'])
        
        return self._time_series_cache[self._time_series_cache['component_id'].isin(component_ids)]
    
    def get_factor_returns_for_factors(self, factor_names: List[str]) -> pd.DataFrame:
        """
        Get factor returns data for specific factors.
        
        Parameters
        ----------
        factor_names : List[str]
            List of factor names to filter for
            
        Returns
        -------
        pd.DataFrame
            Filtered factor returns DataFrame for the specified factors
        """
        if self._factor_returns is None or self._factor_returns.empty:
            return pd.DataFrame(columns=['date', 'factor_name', 'return_value', 'riskmodel_code'])
        
        if not factor_names:
            return pd.DataFrame(columns=['date', 'factor_name', 'return_value', 'riskmodel_code'])
        
        return self._factor_returns[self._factor_returns['factor_name'].isin(factor_names)]
    
    def _setup_risk_service_and_run_analysis(self, force_refresh: bool = False) -> bool:
        """
        Setup risk service with current risk model and run analysis.
        
        Parameters
        ----------
        force_refresh : bool
            Whether to force refresh the analysis
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            if not self.risk_service or not self._current_risk_model:
                return False
            
            # Get factor returns for current risk model
            model_factor_returns = self.load_factor_returns_for_model(self._current_risk_model)
            
            if model_factor_returns.empty:
                print(f"No factor returns found for risk model: {self._current_risk_model}")
                return False
            
            # Convert long format to wide format for risk service
            factor_returns_wide = model_factor_returns.pivot(index='date', columns='factor_name', values='value_column')
            factor_returns_wide.index = pd.to_datetime(factor_returns_wide.index)
            factor_returns_wide.columns.name = None
            
            # Set factor returns and run analysis
            self.risk_service.set_factor_returns(factor_returns_wide)
            analysis_result = self.risk_service.run_risk_analysis(self.root_component_id, force_refresh=force_refresh)
            
            if analysis_result.get('success'):
                self._risk_schema = analysis_result.get('schema')
                # Cache time series data
                self._time_series_cache = self._build_time_series_cache()
                return True
            else:
                print(f"Risk analysis failed for model: {self._current_risk_model}")
                return False
                
        except Exception as e:
            print(f"Error setting up risk service for model {self._current_risk_model}: {e}")
            return False
    
    def get_available_risk_models(self) -> List[Dict[str, str]]:
        """
        Get list of available risk models from configuration or data sources.
        
        Returns
        -------
        List[Dict[str, str]]
            List of available risk models with id, name, and description
        """
        # Return models based on unique riskmodel_codes in factor returns
        if self._factor_returns is not None and 'riskmodel_code' in self._factor_returns.columns:
            unique_models = self._factor_returns['riskmodel_code'].unique()
            return [
                {
                    'id': model_code,
                    'name': model_code.replace('_', ' ').title(),
                    'description': f'{model_code} risk model'
                }
                for model_code in sorted(unique_models)
            ]
        else:
            return []
    
    def load_factor_returns_for_model(self, risk_model_id: str) -> pd.DataFrame:
        """
        Load factor returns for a specific risk model ID.
        
        Parameters
        ----------
        risk_model_id : str
            Risk model identifier
            
        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns: date, factor_name, return_value, riskmodel_code
        """
        if self._factor_returns is None or self._factor_returns.empty:
            return pd.DataFrame(columns=['date', 'factor_name', 'return_value', 'riskmodel_code'])
        
        # Filter factor returns by risk model code
        if 'riskmodel_code' in self._factor_returns.columns:
            model_returns = self._factor_returns[self._factor_returns['riskmodel_code'] == risk_model_id]
            return model_returns.copy()
        else:
            # If no riskmodel_code column, return all data (single model case)
            return self._factor_returns.copy()
    
    def switch_risk_model(self, risk_model_id: str) -> bool:
        """
        Switch to a different risk model and re-run analysis.
        
        Parameters
        ----------
        risk_model_id : str
            Risk model identifier to switch to
            
        Returns
        -------
        bool
            True if switch was successful, False otherwise
        """
        if risk_model_id == self._current_risk_model:
            return True  # Already using this model
            
        # Update current risk model
        self._current_risk_model = risk_model_id
        
        # Setup risk service with new model and run analysis
        return self._setup_risk_service_and_run_analysis(force_refresh=True)
    
    @property
    def current_config_id(self) -> Optional[str]:
        """Get current config ID."""
        return self._current_config_id
    
    @property 
    def current_risk_model(self) -> Optional[str]:
        """Get current risk model ID."""
        return self._current_risk_model
    
    @property
    def root_component_id(self) -> str:
        """Get root component ID from portfolio graph or config."""
        # Priority 1: Use portfolio graph root_id if available
        if self.portfolio_graph and hasattr(self.portfolio_graph, 'root_id') and self.portfolio_graph.root_id:
            return self.portfolio_graph.root_id
        
        # Priority 2: Use config builder_settings root_id
        if self.config and hasattr(self.config, 'builder_settings') and self.config.builder_settings.get('root_id'):
            return self.config.builder_settings['root_id']
        
        # Priority 3: Use first component from config components list
        if self.config and hasattr(self.config, 'components') and self.config.components:
            return self.config.components[0]['path']
        
        # Fallback to 'TOTAL' if nothing found
        return 'TOTAL'