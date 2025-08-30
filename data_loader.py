"""
Modernized Data Loader for Maverick UI

This module provides a clean interface using the simplified risk API
with direct RiskResult access for high-performance risk analysis.
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
    """Modernized data loader using simplified risk API with direct RiskResult access."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.portfolio_graph = None
        self.config = None
        self._current_config_id = None
        self.risk_service = None
        self._factor_returns = None  # Long-format factor returns DataFrame
        
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
                    # Set factor returns and run simplified analysis
                    if self._factor_returns is not None:
                        # Convert factor returns to wide format if needed
                        if 'factor_name' in self._factor_returns.columns:
                            factor_returns_wide = self._factor_returns.pivot(index='date', columns='factor_name', values='return_value')
                        else:
                            factor_returns_wide = self._factor_returns
                        
                        self.risk_service.set_factor_returns(factor_returns_wide)
                        self.risk_service.run_risk_analysis(self.root_component_id)
                
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
            
            # Update risk service and run fresh analysis
            if self.risk_service:
                self.risk_service.set_portfolio_graph(self.portfolio_graph)
                # Set factor returns and run analysis
                if self._factor_returns is not None:
                    # Convert factor returns to wide format if needed
                    if 'factor_name' in self._factor_returns.columns:
                        factor_returns_wide = self._factor_returns.pivot(index='date', columns='factor_name', values='return_value')
                    else:
                        factor_returns_wide = self._factor_returns
                    
                    self.risk_service.set_factor_returns(factor_returns_wide)
                    self.risk_service.run_risk_analysis(self.root_component_id, force_refresh=True)
            
            return True
            
        except Exception as e:
            st.error(f"Failed to load configuration '{config_id}': {str(e)}")
            return False
    
    # =====================================================================
    # CORE DATA ACCESS METHODS
    # =====================================================================
    
    def get_schema_data(self, component_id: str, lens: str) -> Dict[str, Any]:
        """
        Get risk data for a specific component and lens using simplified API.
        
        This is the primary data access method that all UI components should use.
        """
        if not self.risk_service:
            return {}
        
        # Get component analysis from modernized risk service
        result = self.risk_service.get_component_risk_analysis(component_id, lens)
        
        if result.get('success'):
            return result.get('data', {})
        else:
            return {}
    
    def get_all_components(self) -> List[str]:
        """Get list of all component IDs."""
        if not self.portfolio_graph:
            return [self.root_component_id]
        
        return list(self.portfolio_graph.components.keys())
    
    def get_factor_names(self) -> List[str]:
        """Get factor names from multiple sources in priority order."""
        # Priority 1: Extract from risk service factor analysis
        if self.risk_service:
            factor_analysis = self.risk_service.get_factor_analysis()
            if factor_analysis.get('success'):
                factors = factor_analysis.get('factors', [])
                if factors:
                    return factors
        
        # Priority 2: Extract from factor returns DataFrame
        if self._factor_returns is not None and 'factor_name' in self._factor_returns.columns:
            return self._factor_returns['factor_name'].unique().tolist()
        
        # Priority 3: Extract from factor returns columns
        elif self._factor_returns is not None:
            # If factor returns is in wide format, use column names
            return [col for col in self._factor_returns.columns if col != 'date']
        
        # Fallback
        return ['Market', 'Bonds', 'Dollar', 'Commodity', 'Credit']
    
    @property
    def current_config_id(self) -> Optional[str]:
        """Get current config ID."""
        return self._current_config_id
    
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