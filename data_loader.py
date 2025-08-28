"""
Simplified Data Loader using RiskResultSchema as Single Source of Truth

This module provides a minimal interface to the risk analysis system by delegating
all data access to the consolidated schema.py methods, eliminating duplication.
"""

import os
import streamlit as st
from typing import Dict, Any, List, Optional, Union
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
    """Simplified data loader using RiskResultSchema as single source of truth."""
    
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
                    
                    # Run analysis to get schema with hierarchical population
                    try:
                        result = self.risk_service.run_risk_analysis(
                            'TOTAL', 
                            force_refresh=False,
                            populate_hierarchical=True  # NEW: Enable hierarchical population on startup
                        )
                        if result.get('success'):
                            self._risk_schema = result.get('schema')  # Direct schema access
                    except Exception as e:
                        print(f"Risk analysis failed on startup: {e}")
                
        except Exception as e:
            print(f"Could not load default configuration: {e}")
    
    def load_portfolio_configuration(self, config_id: str) -> bool:
        """Load portfolio configuration and run risk analysis with hierarchical population."""
        try:
            from config.portfolio_loader import load_portfolio_from_config_name
            
            config_dir = os.path.join(os.path.dirname(__file__), 'config')
            result = load_portfolio_from_config_name(config_id, config_dir)
            
            self.portfolio_graph = result['portfolio_graph']
            self.factor_returns = result['factor_returns']
            self.config = result['config']
            self._current_config_id = config_id
            
            # Update risk service
            if self.risk_service:
                self.risk_service.set_portfolio_graph(self.portfolio_graph)
                self.risk_service.set_factor_returns(self.factor_returns)
                
                # Run analysis with hierarchical population enabled
                analysis_result = self.risk_service.run_risk_analysis(
                    'TOTAL', 
                    force_refresh=True,
                    populate_hierarchical=True  # NEW: Enable bulk hierarchical storage
                )
                if analysis_result.get('success'):
                    self._risk_schema = analysis_result.get('schema')
            
            return True
            
        except Exception as e:
            st.error(f"Failed to load configuration '{config_id}': {str(e)}")
            return False
    
    # =====================================================================
    # NEW HIERARCHICAL SCHEMA INTEGRATION - Single Source of Truth
    # =====================================================================
    
    def get_comprehensive_schema_data(self, component_id: str = "TOTAL") -> Optional[Dict[str, Any]]:
        """Get comprehensive hierarchical schema data for a component."""
        if not self._risk_schema:
            return None
        return self._risk_schema.data
    
    def get_component_decomposition(self, component_id: str, lens: str) -> Optional[Dict[str, Any]]:
        """Get complete risk decomposition for a specific component and lens."""
        if not self._risk_schema:
            return None
        return self._risk_schema.get_component_decomposition(component_id, lens)
    
    def get_all_component_risk_results(self) -> Dict[str, Dict[str, Any]]:
        """Get risk results for all components."""
        if not self._risk_schema:
            return {}
        return self._risk_schema.get_all_component_risk_results()
    
    def get_component_risk_summary(self, component_id: str) -> Dict[str, Any]:
        """Get comprehensive summary for a component including navigation info."""
        if not self._risk_schema:
            return {'component_id': component_id, 'exists': False}
        return self._risk_schema.get_component_risk_summary(component_id)
    
    def get_hierarchical_validation(self) -> Dict[str, Any]:
        """Get hierarchical data completeness validation."""
        if not self._risk_schema:
            return {'complete': False, 'errors': ['No schema available']}
        return self._risk_schema.validate_hierarchical_completeness()
    
    def get_factor_contributions_from_schema(self, component_id: str, lens: str) -> Dict[str, float]:
        """Get factor contributions for a component from hierarchical schema data."""
        schema_data = self.get_comprehensive_schema_data()
        if not schema_data:
            return {}
        
        hierarchical_data = schema_data.get('hierarchical_risk_data', {})
        component_data = hierarchical_data.get(component_id, {})
        lens_data = component_data.get(lens, {})
        
        if 'decomposer_results' in lens_data:
            return lens_data['decomposer_results'].get('factor_contributions', {})
        elif 'factor_contributions' in lens_data:
            return lens_data['factor_contributions']
        
        return {}
    
    # =====================================================================
    # ENHANCED COMPONENT NAVIGATION
    # =====================================================================
    
    def get_component_hierarchy_path_from_schema(self, component_id: str) -> List[str]:
        """Get hierarchy path using schema navigation methods."""
        if not self._risk_schema:
            return [component_id] if component_id else []
        return self._risk_schema.get_component_hierarchy_path(component_id)
    
    def get_component_parent_from_schema(self, component_id: str) -> Optional[str]:
        """Get parent component using schema methods."""
        if not self._risk_schema:
            return None
        return self._risk_schema.get_component_parent(component_id)
    
    def get_component_children_from_schema(self, component_id: str) -> List[str]:
        """Get child components using schema methods."""
        if not self._risk_schema:
            return []
        return self._risk_schema.get_component_children(component_id)
    
    def get_component_descendants_from_schema(self, component_id: str) -> List[str]:
        """Get all descendant components using schema methods."""
        if not self._risk_schema:
            return []
        return self._risk_schema.get_component_descendants(component_id)
    
    # =====================================================================
    # LEGACY SCHEMA DELEGATION - Maintained for Compatibility
    # =====================================================================
    
    def get_core_metrics(self, lens: str, component_id: str = "TOTAL") -> Dict[str, float]:
        """Get core risk metrics using hierarchical schema integration."""
        # Try hierarchical schema first
        decomposition = self.get_component_decomposition(component_id, lens)
        if decomposition:
            return {
                'total_risk': decomposition.get('total_risk', 0.0),
                'factor_risk': decomposition.get('factor_risk_contribution', 0.0),
                'specific_risk': decomposition.get('specific_risk_contribution', 0.0),
                'tracking_error': decomposition.get('tracking_error', 0.0)
            }
        
        # Fallback to legacy UI metrics if available
        if self._risk_schema:
            return self._risk_schema.get_ui_metrics(component_id, lens)
        return {}
    
    def get_contributions(self, lens: str, contrib_type: str, component_id: str = "TOTAL") -> Dict[str, float]:
        """Get contributions using hierarchical schema integration."""
        # Try hierarchical schema first
        if contrib_type == 'by_factor':
            return self.get_factor_contributions_from_schema(component_id, lens)
        
        # For other contribution types, try decomposition data
        decomposition = self.get_component_decomposition(component_id, lens)
        if decomposition:
            if contrib_type == 'by_asset' and 'asset_contributions' in decomposition:
                return decomposition['asset_contributions']
            elif contrib_type == 'by_component' and 'component_contributions' in decomposition:
                return decomposition['component_contributions']
        
        # Fallback to legacy UI contributions if available
        if self._risk_schema:
            return self._risk_schema.get_ui_contributions(component_id, lens, contrib_type)
        return {}
    
    def get_exposures(self, lens: str, component_id: str = "TOTAL") -> Dict[str, float]:
        """Get factor exposures using hierarchical schema integration."""
        # Try hierarchical schema first
        decomposition = self.get_component_decomposition(component_id, lens)
        if decomposition:
            if 'weighted_betas' in decomposition:
                return decomposition['weighted_betas']
            elif 'factor_exposures' in decomposition:
                return decomposition['factor_exposures']
        
        # Also check exposures section in comprehensive data
        schema_data = self.get_comprehensive_schema_data()
        if schema_data:
            exposures_section = schema_data.get('exposures', {})
            factor_exposures = exposures_section.get('factor_exposures', {})
            if factor_exposures:
                return factor_exposures
        
        # Fallback to legacy UI exposures if available
        if self._risk_schema:
            return self._risk_schema.get_ui_exposures(component_id, lens)
        return {}
    
    def get_weights(self, weight_type: str) -> Dict[str, float]:
        """Get weights using schema delegation."""
        if not self._risk_schema:
            return {}
        weights_data = self._risk_schema.get_ui_weights("TOTAL")
        return weights_data.get(weight_type, {})
    
    def get_matrices(self, matrix_type: str) -> Dict[str, Any]:
        """Get matrix data using schema delegation."""
        if not self._risk_schema:
            return {}
        matrices = self._risk_schema.get_ui_matrices("TOTAL", "portfolio")
        return matrices.get(matrix_type, {})
    
    def get_correlations(self, correlation_type: str) -> Dict[str, Any]:
        """Get correlation data using schema delegation."""
        if not self._risk_schema:
            return {}
        # Access correlations from matrices section
        matrices = self._risk_schema.get_ui_matrices("TOTAL", "portfolio")
        return matrices.get(correlation_type, {})
    
    def to_ui_format(self) -> Dict[str, Any]:
        """Get complete UI-ready data using schema delegation."""
        if not self._risk_schema:
            return self._create_empty_ui_data()
        return self._risk_schema.to_ui_format("TOTAL")
    
    def validate_analysis(self) -> Dict[str, Any]:
        """Validate analysis using hierarchical schema validation."""
        if not self._risk_schema:
            return {"passes": False, "errors": ["No schema available"]}
        
        # Use new hierarchical validation
        hierarchical_validation = self.get_hierarchical_validation()
        
        # Combine with comprehensive validation if available
        comprehensive_validation = {}
        if hasattr(self._risk_schema, 'validate_comprehensive'):
            comprehensive_validation = self._risk_schema.validate_comprehensive()
        
        return {
            "passes": hierarchical_validation.get('complete', False) and comprehensive_validation.get('passes', True),
            "hierarchical_validation": hierarchical_validation,
            "comprehensive_validation": comprehensive_validation,
            "errors": hierarchical_validation.get('errors', []) + comprehensive_validation.get('errors', [])
        }
    
    # =====================================================================
    # UI COMPATIBILITY METHODS - Minimal Implementation
    # =====================================================================
    
    @property
    def data(self) -> Dict[str, Any]:
        """Backward compatibility property."""
        return self.to_ui_format()
    
    def get_factor_names(self) -> List[str]:
        """Get factor names."""
        if self._risk_schema:
            return self._risk_schema.factor_names
        elif self.factor_returns is not None:
            return self.factor_returns.columns.tolist()
        return ['Market', 'Bonds', 'Dollar', 'Commodity', 'Credit']
    
    def get_component_names(self) -> List[str]:
        """Get component names."""
        if self.portfolio_graph:
            return list(self.portfolio_graph.components.keys())
        return ['TOTAL']
    
    def get_available_hierarchical_components(self) -> List[str]:
        """Get hierarchical components."""
        return self.get_component_names()
    
    def get_hierarchy_info(self) -> Dict[str, Any]:
        """Get hierarchy information using hierarchical schema data."""
        schema_data = self.get_comprehensive_schema_data()
        if not schema_data:
            return {'component_metadata': {}}
        
        # Build component metadata from hierarchical data
        hierarchical_data = schema_data.get('hierarchical_risk_data', {})
        component_metadata = {}
        
        for component_id in hierarchical_data.keys():
            summary = self.get_component_risk_summary(component_id)
            navigation = summary.get('navigation', {})
            
            component_metadata[component_id] = {
                'name': component_id,  # Could be enhanced with actual component names
                'type': 'leaf' if not navigation.get('children') else 'node',
                'level': len(self.get_component_hierarchy_path_from_schema(component_id)) - 1,
                'parent': navigation.get('parent'),
                'children': navigation.get('children', []),
                'risk_data_available': summary.get('exists', False)
            }
        
        return {
            'component_metadata': component_metadata,
            'hierarchy_structure': schema_data.get('hierarchy', {})
        }
    
    def get_component_hierarchy_path(self, component_id: str) -> List[str]:
        """Get hierarchy path using schema methods."""
        return self.get_component_hierarchy_path_from_schema(component_id)
    
    def can_drill_up(self, component_id: str) -> bool:
        """Check if can drill up."""
        return component_id != 'TOTAL' and '/' in component_id
    
    def get_component_parent(self, component_id: str) -> Optional[str]:
        """Get parent component using schema methods."""
        return self.get_component_parent_from_schema(component_id)
    
    def get_drilldown_options(self, component_id: str) -> List[str]:
        """Get drill-down options using schema methods."""
        return self.get_component_children_from_schema(component_id)
    
    def get_component_lens_availability(self, component_id: str) -> List[str]:
        """Get available lenses."""
        return ['portfolio', 'benchmark', 'active']
    
    def get_component_validation_status(self, component_id: str, lens: str) -> Dict[str, bool]:
        """Get validation status."""
        validation = self.validate_analysis()
        return {'euler_identity_check': validation.get('checks', {}).get('euler_identity', True)}
    
    def get_time_series_data(self, metric_name: str, component_id: str) -> List[float]:
        """Get time series data."""
        ui_data = self.to_ui_format()
        time_series = ui_data.get('time_series', {})
        
        if metric_name in time_series:
            ts_data = time_series[metric_name]
            if component_id in ts_data:
                return ts_data[component_id]
        
        # Mock data for compatibility
        import numpy as np
        np.random.seed(42)
        return np.random.normal(0.0005, 0.01, 252).tolist()
    
    def get_validation_info(self) -> Dict[str, Any]:
        """Get validation info including hierarchical completeness."""
        validation_result = self.validate_analysis()
        hierarchical_validation = self.get_hierarchical_validation()
        
        return {
            **validation_result,
            'hierarchical_summary': {
                'total_components': hierarchical_validation.get('total_components', 0),
                'components_with_data': hierarchical_validation.get('components_with_data', 0),
                'completeness_percentage': hierarchical_validation.get('summary', {}).get('completeness_percentage', 0),
                'lens_coverage': hierarchical_validation.get('lens_coverage', {})
            }
        }
    
    def get_hierarchical_data_summary(self) -> Dict[str, Any]:
        """Get hierarchical summary using actual schema data."""
        hierarchical_validation = self.get_hierarchical_validation()
        all_components = self.get_all_component_risk_results()
        
        # Calculate lens coverage per component
        component_lens_counts = {}
        for comp_id, comp_data in all_components.items():
            component_lens_counts[comp_id] = len(comp_data.keys())
        
        return {
            'total_components': hierarchical_validation.get('total_components', 0),
            'components_with_data': hierarchical_validation.get('components_with_data', 0),
            'components_with_matrices': len([comp for comp in all_components if all_components[comp]]),
            'schema_version': '4.0',  # Updated to reflect hierarchical schema
            'component_lens_counts': component_lens_counts,
            'lens_coverage': hierarchical_validation.get('lens_coverage', {}),
            'completeness_summary': hierarchical_validation.get('summary', {})
        }
    
    def has_risk_service(self) -> bool:
        """Check if risk service is available."""
        return self.risk_service is not None
    
    def refresh_data(self):
        """Refresh data."""
        if self._current_config_id:
            self.load_portfolio_configuration(self._current_config_id)
    
    def get_risk_analysis_status(self) -> Dict[str, Any]:
        """Get risk analysis status."""
        if not self.risk_service:
            return {'available': False, 'message': 'Risk service not available'}
        
        return {
            'available': True,
            'ready_for_analysis': True,
            'analysis_completed': self._risk_schema is not None,
            'portfolio_components': len(self.portfolio_graph.components) if self.portfolio_graph else 0,
            'factor_count': len(self.factor_returns.columns) if self.factor_returns is not None else 0,
            'cache_size': 0,
            'last_analysis': self._risk_schema.timestamp.isoformat() if self._risk_schema else None
        }
    
    def run_risk_analysis(self, component_id: str = None, force_refresh: bool = False) -> bool:
        """Run risk analysis."""
        if not self.risk_service:
            return False
        
        try:
            result = self.risk_service.run_risk_analysis(
                root_component_id=component_id or 'TOTAL',
                force_refresh=force_refresh
            )
            if result.get('success'):
                self._risk_schema = result.get('schema')
                return True
            return False
        except:
            return False
    
    def load_risk_model(self, model_id: str) -> bool:
        """Load risk model."""
        # Placeholder for compatibility - model_id parameter kept for interface compatibility
        _ = model_id  # Acknowledge parameter to avoid warnings
        return True
    
    def _auto_run_risk_analysis(self):
        """Auto-run risk analysis."""
        if self.risk_service:
            self.run_risk_analysis()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information for data management tab."""
        return {
            'cache_size': 0,
            'cache_entries': 0,
            'last_refresh': None,
            'cache_hit_rate': 0.0
        }
    
    def _create_empty_ui_data(self) -> Dict[str, Any]:
        """Create empty UI data structure."""
        return {
            'metadata': {'analysis_type': 'Unknown', 'schema_version': '3.0'},
            'core_metrics': {},
            'portfolio': {'core_metrics': {}, 'contributions': {}, 'exposures': {}},
            'benchmark': {'core_metrics': {}, 'contributions': {}, 'exposures': {}},
            'active': {'core_metrics': {}, 'contributions': {}, 'exposures': {}},
            'weights': {},
            'hierarchy': {},
            'time_series': {},
            'validation': {'passes': False},
            'identifiers': {'factor_names': self.get_factor_names()}
        }
    
    @property
    def current_config_id(self) -> Optional[str]:
        """Get current config ID."""
        return self._current_config_id