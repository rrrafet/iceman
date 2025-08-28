"""
Minimal Schema-Only Data Loader for Maverick UI

This module provides a radically simplified interface that uses only the
RiskResultSchema from the schema factory as the single source of truth.
All complex legacy data transformation and mapping logic has been removed.
"""

import os
import streamlit as st
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class SidebarState:
    """State object containing sidebar filter selections."""
    lens: str
    selected_node: str
    date_range: tuple
    selected_factors: List[str]
    annualized: bool
    show_percentage: bool


class DataLoader:
    """Minimal data loader using RiskResultSchema as single source of truth."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.portfolio_graph = None
        self.factor_returns = None
        self.config = None
        self._current_config_id = None
        self.risk_service = None
        self._risk_schema = None  # Single source of truth
        
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
                self.factor_returns = result['factor_returns']
                self.config = result['config']
                self._current_config_id = default_config
                
                # Initialize risk service
                self.risk_service = create_risk_service()
                if self.risk_service:
                    self.risk_service.set_portfolio_graph(self.portfolio_graph)
                    self.risk_service.set_factor_returns(self.factor_returns)
                    
                    # Run analysis to get schema
                    try:
                        result = self.risk_service.run_risk_analysis('TOTAL', force_refresh=False)
                        if result.get('success'):
                            self._risk_schema = result.get('schema')
                    except Exception as e:
                        print(f"Risk analysis failed on startup: {e}")
                
        except Exception as e:
            print(f"Could not load default configuration: {e}")
    
    def load_portfolio_configuration(self, config_id: str) -> bool:
        """Load portfolio configuration and run risk analysis."""
        try:
            from config.portfolio_loader import load_portfolio_from_config_name
            
            config_dir = os.path.join(os.path.dirname(__file__), 'config')
            result = load_portfolio_from_config_name(config_id, config_dir)
            
            self.portfolio_graph = result['portfolio_graph']
            self.factor_returns = result['factor_returns']
            self.config = result['config']
            self._current_config_id = config_id
            
            # Update risk service and get fresh schema
            if self.risk_service:
                self.risk_service.set_portfolio_graph(self.portfolio_graph)
                self.risk_service.set_factor_returns(self.factor_returns)
                
                analysis_result = self.risk_service.run_risk_analysis('TOTAL', force_refresh=True)
                if analysis_result.get('success'):
                    self._risk_schema = analysis_result.get('schema')
            
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
            return ['TOTAL']
        
        return self._risk_schema.get_all_components()
    
    def get_factor_names(self) -> List[str]:
        """Get factor names."""
        if self._risk_schema:
            return self._risk_schema.factor_names
        elif self.factor_returns is not None:
            return self.factor_returns.columns.tolist()
        return ['Market', 'Bonds', 'Dollar', 'Commodity', 'Credit']
    
    @property
    def current_config_id(self) -> Optional[str]:
        """Get current config ID."""
        return self._current_config_id