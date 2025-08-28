"""
Risk Schema Factory - Factory methods for creating RiskResultSchema from various data sources.

This module implements the hierarchical extraction method for comprehensive data
collection from FactorRiskDecompositionVisitor objects and other risk data sources.
"""

from typing import Dict, List, Any, Optional, Set
import numpy as np
import pandas as pd
import logging

from .schema import RiskResultSchema
from .schema_utils import (
    array_to_named_dict, matrix_to_nested_dict, extract_property_safely,
    validate_array_dimensions, process_time_series_data, calculate_risk_percentages,
    validate_weight_vector, merge_nested_dicts, clean_nan_values
)

logger = logging.getLogger(__name__)


class RiskSchemaFactory:
    """
    Factory class for creating RiskResultSchema instances from various data sources.
    
    Implements the hierarchical extraction method as specified in the requirements,
    providing comprehensive data collection from portfolio visitors with full
    error handling and graceful degradation.
    """
    
    @classmethod
    def from_factor_risk_decomposition_visitor(
        cls,
        visitor,
        root_component_id: str,
        map_full_hierarchy: bool = True
    ) -> RiskResultSchema:
        """
        Create schema from FactorRiskDecompositionVisitor using hierarchical extraction.
        
        This is the primary factory method implementing the comprehensive hierarchical
        extraction system that collects 36+ properties per component per lens.
        
        Parameters
        ----------
        visitor : FactorRiskDecompositionVisitor
            Processed visitor containing risk decomposition results
        root_component_id : str
            Root component to start extraction from
        map_full_hierarchy : bool, default True
            Whether to extract data from all processed components
            
        Returns
        -------
        RiskResultSchema
            Populated schema with hierarchical risk data
        """
        logger.info(f"Starting hierarchical extraction from visitor for root: {root_component_id}")
        
        # Extract global metadata
        factor_names = cls._extract_factor_names(visitor)
        asset_names_global = cls._extract_global_asset_names(visitor, root_component_id)
        
        logger.debug(f"Extracted global metadata: {len(factor_names)} factors, {len(asset_names_global)} assets")
        
        # Create base schema
        schema = RiskResultSchema(
            analysis_type="hierarchical",
            extraction_method="direct_mapping",
            metadata={
                'visitor_type': type(visitor).__name__,
                'root_component_id': root_component_id,
                'mapping_mode': 'full_hierarchy' if map_full_hierarchy else 'single_component'
            }
        )
        
        # Perform hierarchical tree mapping
        hierarchical_data = cls._map_hierarchical_tree(
            visitor, root_component_id, map_full_hierarchy, asset_names_global, factor_names
        )
        
        # Set hierarchical data in schema
        schema.set_hierarchical_risk_data(hierarchical_data)
        
        # Extract and set time series data
        time_series_data = cls._extract_time_series_from_visitor(visitor, hierarchical_data)
        schema.set_time_series_data(time_series_data)
        
        # Extract factor analysis
        factor_analysis = cls._extract_factor_analysis(hierarchical_data, factor_names)
        schema.set_factor_analysis(factor_analysis)
        
        # Set global metadata
        global_metadata = {
            'factor_names': factor_names,
            'asset_names': asset_names_global,
            'components_mapped': len(hierarchical_data),
            'extraction_timestamp': pd.Timestamp.now().isoformat()
        }
        schema.set_global_metadata(global_metadata)
        
        logger.info(f"Hierarchical extraction completed: {len(hierarchical_data)} components mapped")
        return schema
    
    @classmethod
    def from_visitor_direct_mapping(
        cls,
        visitor,
        root_component_id: str,
        map_full_hierarchy: bool = True
    ) -> RiskResultSchema:
        """
        Alternative factory method using direct mapping from visitor metric store.
        
        Parameters
        ----------
        visitor : FactorRiskDecompositionVisitor
            Processed visitor containing risk decomposition results
        root_component_id : str
            Root component to start extraction from
        map_full_hierarchy : bool, default True
            Whether to extract data from all processed components
            
        Returns
        -------
        RiskResultSchema
            Populated schema with hierarchical risk data
        """
        return cls.from_factor_risk_decomposition_visitor(
            visitor, root_component_id, map_full_hierarchy
        )
    
    @classmethod
    def _extract_factor_names(cls, visitor) -> List[str]:
        """Extract factor names from visitor."""
        try:
            if hasattr(visitor, 'factor_names') and visitor.factor_names:
                return list(visitor.factor_names)
            elif hasattr(visitor, 'get_factor_names'):
                return visitor.get_factor_names()
            elif hasattr(visitor, 'factor_returns') and hasattr(visitor.factor_returns, 'columns'):
                return list(visitor.factor_returns.columns)
            else:
                logger.warning("Could not extract factor names from visitor")
                return []
        except Exception as e:
            logger.error(f"Error extracting factor names: {e}")
            return []
    
    @classmethod
    def _extract_global_asset_names(cls, visitor, root_component_id: str) -> List[str]:
        """Extract global asset names from visitor for the root component."""
        try:
            if hasattr(visitor, 'metric_store') and visitor.metric_store:
                asset_names_metric = visitor.metric_store.get_metric(root_component_id, 'asset_names')
                if asset_names_metric:
                    return list(asset_names_metric.value())
            
            # Fallback: try to get from processed components
            if hasattr(visitor, 'get_processed_components'):
                components = visitor.get_processed_components()
                return [comp for comp in components if not '/' in comp]  # Leaf components
            
            logger.warning("Could not extract asset names from visitor")
            return []
        except Exception as e:
            logger.error(f"Error extracting asset names: {e}")
            return []
    
    @classmethod
    def _map_hierarchical_tree(
        cls,
        visitor,
        root_component_id: str,
        map_full_hierarchy: bool,
        asset_names: List[str],
        factor_names: List[str]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Perform comprehensive traversal of the portfolio hierarchy.
        
        Discovery Phase: Identify all processed components
        Data Collection Phase: Extract comprehensive data from each component
        """
        hierarchical_data = {}
        
        try:
            # Discovery Phase: Component Discovery
            if map_full_hierarchy:
                processed_components = cls._get_processed_components(visitor)
            else:
                processed_components = {root_component_id}
            
            logger.info(f"Discovered {len(processed_components)} components to process")
            
            # Data Collection Phase: Per-Component Extraction
            for component_id in processed_components:
                logger.debug(f"Processing component: {component_id}")
                
                # Get component-specific asset names
                component_asset_names = cls._get_component_asset_names(visitor, component_id) or asset_names
                
                # Extract comprehensive node risk data
                node_data = cls._extract_node_risk_data(
                    visitor, component_id, component_asset_names, factor_names
                )
                
                if node_data:
                    hierarchical_data[component_id] = node_data
                    logger.debug(f"Successfully extracted data for {component_id}: {list(node_data.keys())} lenses")
                else:
                    logger.warning(f"No data extracted for component {component_id}")
        
        except Exception as e:
            logger.error(f"Error in hierarchical tree mapping: {e}")
            # Return partial data if available
        
        return hierarchical_data
    
    @classmethod
    def _get_processed_components(cls, visitor) -> Set[str]:
        """Get all processed components from visitor using multiple strategies."""
        components = set()
        
        try:
            # Strategy 1: Direct method if available
            if hasattr(visitor, 'get_processed_components'):
                components.update(visitor.get_processed_components())
                logger.debug(f"Found {len(components)} components via get_processed_components()")
            
            # Strategy 2: Check metric store for components with risk models
            if hasattr(visitor, 'metric_store') and visitor.metric_store:
                # Look for components with hierarchical_model_context
                all_metrics = getattr(visitor.metric_store, '_metrics', {})
                for comp_id in all_metrics.keys():
                    context_metric = visitor.metric_store.get_metric(comp_id, 'hierarchical_model_context')
                    if context_metric:
                        components.add(comp_id)
            
            # Strategy 3: Check internal visitor storage
            if hasattr(visitor, '_processed_components'):
                components.update(visitor._processed_components)
            
            if hasattr(visitor, '_unified_matrices'):
                components.update(visitor._unified_matrices.keys())
            
        except Exception as e:
            logger.error(f"Error getting processed components: {e}")
        
        return components
    
    @classmethod
    def _get_component_asset_names(cls, visitor, component_id: str) -> Optional[List[str]]:
        """Get asset names specific to a component."""
        try:
            if hasattr(visitor, 'metric_store') and visitor.metric_store:
                asset_names_metric = visitor.metric_store.get_metric(component_id, 'asset_names')
                if asset_names_metric:
                    return list(asset_names_metric.value())
        except Exception as e:
            logger.debug(f"Could not get asset names for {component_id}: {e}")
        return None
    
    @classmethod
    def _extract_node_risk_data(
        cls,
        visitor,
        component_id: str,
        asset_names: List[str],
        factor_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract comprehensive data from an individual component.
        
        Uses metric store access strategy with hierarchical model context for nodes
        and individual risk models for leaves.
        """
        node_data = {}
        
        try:
            # Primary Path: Hierarchical model context for node components
            context = cls._get_hierarchical_context(visitor, component_id)
            if context:
                logger.debug(f"Found hierarchical context for {component_id}")
                
                # Extract data for all lenses using hierarchical context
                for lens_name in ['portfolio', 'benchmark', 'active']:
                    lens_data = cls._extract_lens_data_from_context(
                        context, lens_name, asset_names, factor_names
                    )
                    if lens_data:
                        node_data[lens_name] = lens_data
                
            else:
                # Secondary Path: Individual risk models for leaf components
                logger.debug(f"No hierarchical context found for {component_id}, trying individual models")
                node_data = cls._extract_from_individual_models(
                    visitor, component_id, asset_names, factor_names
                )
            
            # Extract time series data from visitor metric store for this component
            component_time_series = cls._extract_component_time_series(visitor, component_id)
            if component_time_series:
                # Add time series to each lens data
                for lens_name in ['portfolio', 'benchmark', 'active']:
                    if lens_name in node_data:
                        lens_time_series = component_time_series.get(lens_name, {})
                        if lens_time_series:
                            node_data[lens_name].update(lens_time_series)
        
        except Exception as e:
            logger.error(f"Error extracting node data for {component_id}: {e}")
        
        return node_data
    
    @classmethod
    def _get_hierarchical_context(cls, visitor, component_id: str):
        """Get hierarchical model context from metric store."""
        try:
            if hasattr(visitor, 'metric_store') and visitor.metric_store:
                context_metric = visitor.metric_store.get_metric(component_id, 'hierarchical_model_context')
                if context_metric:
                    return context_metric.value()
        except Exception as e:
            logger.debug(f"Could not get hierarchical context for {component_id}: {e}")
        return None
    
    @classmethod
    def _extract_lens_data_from_context(
        cls,
        context,
        lens_name: str,
        asset_names: List[str],
        factor_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Extract comprehensive risk data for a specific lens from hierarchical context."""
        try:
            # Get the decomposer for this lens
            decomposer = getattr(context, f'{lens_name}_decomposer', None)
            if not decomposer:
                logger.debug(f"No {lens_name} decomposer found in context")
                return None
            
            # Extract comprehensive decomposer data
            lens_data = cls._extract_decomposer_data(decomposer, asset_names, factor_names)
            
            # Extract time series data from decomposer
            time_series_data = cls._extract_time_series_from_decomposer(decomposer, lens_name)
            if time_series_data:
                lens_data.update(time_series_data)
            
            # Add lens-specific metadata
            lens_data['lens_type'] = lens_name
            lens_data['extraction_method'] = 'hierarchical_context'
            
            return lens_data
        
        except Exception as e:
            logger.error(f"Error extracting {lens_name} lens data: {e}")
            return None
    
    @classmethod
    def _extract_from_individual_models(
        cls,
        visitor,
        component_id: str,
        asset_names: List[str],
        factor_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract data from individual risk models (for leaf components)."""
        node_data = {}
        
        try:
            if hasattr(visitor, 'metric_store') and visitor.metric_store:
                # Try to find individual risk models
                for lens_name in ['portfolio', 'benchmark', 'active']:
                    model_metric = visitor.metric_store.get_metric(component_id, f'{lens_name}_risk_model')
                    if model_metric:
                        # Create a simple decomposer context and extract data
                        # This would require creating a single model context
                        logger.debug(f"Found individual {lens_name} model for {component_id}")
                        # Note: This would require integration with RiskDecomposer
                        # For now, create basic structure
                        node_data[lens_name] = {
                            'lens_type': lens_name,
                            'extraction_method': 'individual_model',
                            'total_risk': 0.0,
                            'factor_contributions': {},
                            'asset_contributions': {}
                        }
        
        except Exception as e:
            logger.error(f"Error extracting individual models for {component_id}: {e}")
        
        return node_data
    
    @classmethod
    def _extract_decomposer_data(
        cls,
        decomposer,
        asset_names: List[str],
        factor_names: List[str]
    ) -> Dict[str, Any]:
        """
        Extract comprehensive data from a decomposer instance.
        
        Performs exhaustive property extraction of 36+ properties including:
        - Core risk metrics
        - Array properties converted to named dictionaries
        - Advanced properties (marginal contributions, percentages)
        - Active risk properties (when available)
        - Matrix data
        """
        data = {}
        
        try:
            # Core Risk Metrics
            data['total_risk'] = extract_property_safely(decomposer, 'portfolio_volatility', 0.0)
            data['factor_risk_contribution'] = extract_property_safely(decomposer, 'factor_risk_contribution', 0.0)
            data['specific_risk_contribution'] = extract_property_safely(decomposer, 'specific_risk_contribution', 0.0)
            
            # Calculate risk percentages
            risk_percentages = calculate_risk_percentages(
                data['total_risk'], data['factor_risk_contribution'], data['specific_risk_contribution']
            )
            data.update(risk_percentages)
            
            # Array Properties � Named Dictionaries
            
            # Factor contributions
            factor_contrib_array = extract_property_safely(decomposer, 'factor_contributions')
            if factor_contrib_array is not None:
                data['factor_contributions'] = array_to_named_dict(
                    factor_contrib_array, factor_names, "Factor"
                )
            else:
                data['factor_contributions'] = {}
            
            # Asset contributions
            asset_contrib_array = extract_property_safely(decomposer, 'asset_total_contributions')
            if asset_contrib_array is not None:
                data['asset_contributions'] = array_to_named_dict(
                    asset_contrib_array, asset_names, "Asset"
                )
            else:
                data['asset_contributions'] = {}
            
            # Portfolio factor exposures
            exposure_array = extract_property_safely(decomposer, 'portfolio_factor_exposure')
            if exposure_array is not None:
                data['factor_exposures'] = array_to_named_dict(
                    exposure_array, factor_names, "Factor"
                )
            else:
                data['factor_exposures'] = {}
            
            # Portfolio weights
            weights_array = extract_property_safely(decomposer, 'portfolio_weights')
            if weights_array is not None:
                data['portfolio_weights'] = array_to_named_dict(
                    weights_array, asset_names, "Asset"
                )
                
                # Validate weights
                weight_validation = validate_weight_vector(weights_array, 'portfolio')
                data['weight_validation'] = weight_validation
            else:
                data['portfolio_weights'] = {}
            
            # Advanced Properties
            
            # Marginal contributions
            marginal_factor = extract_property_safely(decomposer, 'marginal_factor_contributions')
            if marginal_factor is not None:
                data['marginal_factor_contributions'] = array_to_named_dict(
                    marginal_factor, factor_names, "Factor"
                )
            
            marginal_asset = extract_property_safely(decomposer, 'marginal_asset_contributions')
            if marginal_asset is not None:
                data['marginal_asset_contributions'] = array_to_named_dict(
                    marginal_asset, asset_names, "Asset"
                )
            
            # Percentage contributions
            percent_total = extract_property_safely(decomposer, 'percent_total_contributions')
            if percent_total is not None:
                data['percent_total_contributions'] = array_to_named_dict(
                    percent_total, asset_names, "Asset"
                )
            
            percent_factor = extract_property_safely(decomposer, 'percent_factor_contributions')
            if percent_factor is not None:
                data['percent_factor_contributions'] = array_to_named_dict(
                    percent_factor, factor_names, "Factor"
                )
            
            # Active Risk Properties (when available)
            
            # Check if this is an active risk decomposer
            if hasattr(decomposer, 'total_active_risk'):
                data['total_active_risk'] = extract_property_safely(decomposer, 'total_active_risk', 0.0)
                
                # Benchmark weights
                bench_weights = extract_property_safely(decomposer, 'benchmark_weights')
                if bench_weights is not None:
                    data['benchmark_weights'] = array_to_named_dict(
                        bench_weights, asset_names, "Asset"
                    )
                
                # Active weights  
                active_weights = extract_property_safely(decomposer, 'active_weights')
                if active_weights is not None:
                    data['active_weights'] = array_to_named_dict(
                        active_weights, asset_names, "Asset"
                    )
                
                # Allocation and selection risk
                data['allocation_factor_risk'] = extract_property_safely(decomposer, 'allocation_factor_risk', 0.0)
                data['selection_factor_risk'] = extract_property_safely(decomposer, 'selection_factor_risk', 0.0)
                data['total_allocation_risk'] = extract_property_safely(decomposer, 'total_allocation_risk', 0.0)
                data['total_selection_risk'] = extract_property_safely(decomposer, 'total_selection_risk', 0.0)
            
            # Matrix Data
            
            # Asset by factor contributions matrix (NxK matrix � nested dictionary)
            asset_factor_matrix = extract_property_safely(decomposer, 'asset_by_factor_contributions')
            if asset_factor_matrix is not None:
                data['asset_by_factor_matrix'] = matrix_to_nested_dict(
                    asset_factor_matrix, asset_names, factor_names, "Asset", "Factor"
                )
            
            # Weighted betas
            weighted_betas = extract_property_safely(decomposer, 'weighted_betas')
            if weighted_betas is not None:
                if isinstance(weighted_betas, dict):
                    data['weighted_betas'] = weighted_betas
                else:
                    data['weighted_betas'] = array_to_named_dict(
                        weighted_betas, factor_names, "Factor"
                    )
            
            # Additional matrix properties if available
            for prop_name in ['factor_covariance', 'residual_covariance', 'correlation_matrix']:
                prop_value = extract_property_safely(decomposer, prop_name)
                if prop_value is not None:
                    data[prop_name] = prop_value
            
            # Add decomposer results summary if available
            if hasattr(decomposer, 'risk_decomposition_summary'):
                try:
                    summary = decomposer.risk_decomposition_summary()
                    if isinstance(summary, dict):
                        data['decomposer_results'] = summary
                except Exception as e:
                    logger.debug(f"Could not extract decomposer summary: {e}")
        
        except Exception as e:
            logger.error(f"Error extracting decomposer data: {e}")
        
        # Clean any NaN values
        data = clean_nan_values(data)
        
        return data
    
    @classmethod
    def _extract_time_series_from_visitor(
        cls,
        visitor,
        hierarchical_data: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> Dict[str, pd.DataFrame]:
        """Extract time series data from visitor and hierarchical data."""
        time_series = {}
        
        try:
            # Extract dates from visitor
            dates = cls._extract_dates_from_visitor(visitor)
            if dates is None:
                logger.warning("Could not extract dates for time series")
                return {}
            
            # Extract factor returns
            if hasattr(visitor, 'factor_returns') and isinstance(visitor.factor_returns, pd.DataFrame):
                time_series['factor_returns'] = visitor.factor_returns.copy()
            
            # Extract component-level time series from metric store
            if hasattr(visitor, 'metric_store') and visitor.metric_store:
                for component_id in hierarchical_data.keys():
                    # Portfolio returns
                    port_metric = visitor.metric_store.get_metric(component_id, 'portfolio_return')
                    if port_metric and hasattr(port_metric.value(), 'index'):
                        time_series[f'{component_id}_portfolio_returns'] = port_metric.value().to_frame(component_id)
                    
                    # Benchmark returns
                    bench_metric = visitor.metric_store.get_metric(component_id, 'benchmark_return')  
                    if bench_metric and hasattr(bench_metric.value(), 'index'):
                        time_series[f'{component_id}_benchmark_returns'] = bench_metric.value().to_frame(f'{component_id}_benchmark')
                    
                    # Active returns (computed)
                    if port_metric and bench_metric:
                        try:
                            port_returns = port_metric.value()
                            bench_returns = bench_metric.value()
                            active_returns = port_returns - bench_returns
                            time_series[f'{component_id}_active_returns'] = active_returns.to_frame(f'{component_id}_active')
                        except Exception as e:
                            logger.debug(f"Could not compute active returns for {component_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error extracting time series data: {e}")
        
        return time_series
    
    @classmethod
    def _extract_time_series_from_decomposer(
        cls,
        decomposer,
        lens_name: str
    ) -> Dict[str, Any]:
        """Extract time series data from a decomposer instance."""
        time_series_data = {}
        
        try:
            # Try to extract time series data from decomposer
            # Look for common time series properties
            time_series_properties = [
                'portfolio_returns',
                'benchmark_returns', 
                'active_returns',
                'returns',
                'portfolio_time_series',
                'time_series'
            ]
            
            dates = None
            
            for prop_name in time_series_properties:
                prop_value = extract_property_safely(decomposer, prop_name)
                if prop_value is not None:
                    # Extract dates from pandas Series if available
                    if hasattr(prop_value, 'index') and dates is None:
                        dates = prop_value.index.tolist()
                    
                    # Convert values to list
                    if hasattr(prop_value, 'tolist'):
                        values = prop_value.tolist()
                    elif hasattr(prop_value, 'values'):
                        values = prop_value.values.tolist()
                    elif isinstance(prop_value, (list, tuple)):
                        values = list(prop_value)
                    else:
                        values = prop_value
                    
                    # Store both values and dates
                    time_series_data[prop_name] = values
                    if dates is not None and len(dates) == len(values):
                        time_series_data[f'{prop_name}_dates'] = dates
            
            # Add generic 'returns' and 'dates' fields based on lens name
            if lens_name == 'portfolio' and 'portfolio_returns' in time_series_data:
                time_series_data['returns'] = time_series_data['portfolio_returns']
                if 'portfolio_returns_dates' in time_series_data:
                    time_series_data['dates'] = time_series_data['portfolio_returns_dates']
            elif lens_name == 'benchmark' and 'benchmark_returns' in time_series_data:
                time_series_data['returns'] = time_series_data['benchmark_returns']
                if 'benchmark_returns_dates' in time_series_data:
                    time_series_data['dates'] = time_series_data['benchmark_returns_dates']
            elif lens_name == 'active' and 'active_returns' in time_series_data:
                time_series_data['returns'] = time_series_data['active_returns']
                if 'active_returns_dates' in time_series_data:
                    time_series_data['dates'] = time_series_data['active_returns_dates']
            
            logger.debug(f"Extracted {len(time_series_data)} time series properties from {lens_name} decomposer")
            
        except Exception as e:
            logger.debug(f"Error extracting time series from {lens_name} decomposer: {e}")
        
        return time_series_data
    
    @classmethod 
    def _extract_component_time_series(
        cls,
        visitor,
        component_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """Extract time series data for a specific component from visitor metric store."""
        component_time_series = {}
        
        try:
            if hasattr(visitor, 'metric_store') and visitor.metric_store:
                # Define time series metric patterns for each lens
                time_series_patterns = {
                    'portfolio': ['portfolio_return', 'portfolio_returns', 'return'],
                    'benchmark': ['benchmark_return', 'benchmark_returns', 'benchmark'],
                    'active': ['active_return', 'active_returns', 'excess_return']
                }
                
                for lens_name, patterns in time_series_patterns.items():
                    lens_time_series = {}
                    
                    for pattern in patterns:
                        metric = visitor.metric_store.get_metric(component_id, pattern)
                        if metric:
                            try:
                                metric_value = metric.value()
                                
                                # Extract dates if available
                                dates = None
                                if hasattr(metric_value, 'index'):
                                    dates = metric_value.index.tolist()
                                
                                # Convert values to list
                                if hasattr(metric_value, 'tolist'):
                                    time_series_list = metric_value.tolist()
                                elif hasattr(metric_value, 'values'):
                                    time_series_list = metric_value.values.tolist()
                                elif isinstance(metric_value, (list, tuple)):
                                    time_series_list = list(metric_value)
                                else:
                                    continue  # Skip if we can't convert to list
                                
                                # Store with multiple keys for compatibility
                                if pattern.endswith('_return') or pattern.endswith('_returns'):
                                    lens_time_series[pattern] = time_series_list
                                    if pattern.endswith('s'):
                                        lens_time_series[pattern[:-1]] = time_series_list
                                    else:
                                        lens_time_series[pattern + 's'] = time_series_list
                                    
                                    # Store dates alongside values
                                    if dates is not None and len(dates) == len(time_series_list):
                                        lens_time_series[f'{pattern}_dates'] = dates
                                        if pattern.endswith('s'):
                                            lens_time_series[f'{pattern[:-1]}_dates'] = dates
                                        else:
                                            lens_time_series[f'{pattern}s_dates'] = dates
                                
                                # Add generic 'returns' and 'dates' fields for this lens
                                if 'returns' not in lens_time_series:
                                    lens_time_series['returns'] = time_series_list
                                    if dates is not None and len(dates) == len(time_series_list):
                                        lens_time_series['dates'] = dates
                                
                                logger.debug(f"Extracted {pattern} time series for {component_id}/{lens_name}: {len(time_series_list)} points")
                                break  # Use first pattern that works
                                
                            except Exception as e:
                                logger.debug(f"Could not extract {pattern} for {component_id}: {e}")
                                continue
                    
                    if lens_time_series:
                        component_time_series[lens_name] = lens_time_series
                        
        except Exception as e:
            logger.debug(f"Error extracting component time series for {component_id}: {e}")
        
        return component_time_series
    
    @classmethod
    def _extract_dates_from_visitor(cls, visitor) -> Optional[pd.Index]:
        """Extract date information from visitor using multiple sources."""
        try:
            # Try direct visitor attributes
            if hasattr(visitor, 'dates') and visitor.dates is not None:
                return visitor.dates
            
            # Try context attributes
            if hasattr(visitor, 'context') and hasattr(visitor.context, 'dates'):
                return visitor.context.dates
            
            # Try factor returns index
            if hasattr(visitor, 'factor_returns') and hasattr(visitor.factor_returns, 'index'):
                return visitor.factor_returns.index
            
            # Try metric store time series data
            if hasattr(visitor, 'metric_store') and visitor.metric_store:
                # Look for any time series metric
                all_metrics = getattr(visitor.metric_store, '_metrics', {})
                for comp_id, metrics in all_metrics.items():
                    for metric_name, metric in metrics.items():
                        if 'return' in metric_name and hasattr(metric.value(), 'index'):
                            return metric.value().index
        
        except Exception as e:
            logger.debug(f"Error extracting dates: {e}")
        
        return None
    
    @classmethod
    def _extract_factor_analysis(
        cls,
        hierarchical_data: Dict[str, Dict[str, Dict[str, Any]]],
        factor_names: List[str]
    ) -> Dict[str, Any]:
        """Extract aggregated factor analysis from hierarchical data."""
        factor_analysis = {
            'factors': factor_names,
            'factor_count': len(factor_names),
            'components_analyzed': len(hierarchical_data)
        }
        
        try:
            # Aggregate factor contributions across all components (portfolio lens)
            aggregated_contributions = {}
            component_count = 0
            
            for component_id, component_data in hierarchical_data.items():
                portfolio_data = component_data.get('portfolio', {})
                factor_contributions = portfolio_data.get('factor_contributions', {})
                
                if factor_contributions:
                    component_count += 1
                    for factor, contribution in factor_contributions.items():
                        if factor not in aggregated_contributions:
                            aggregated_contributions[factor] = []
                        aggregated_contributions[factor].append(contribution)
            
            # Calculate average contributions
            if aggregated_contributions:
                avg_contributions = {
                    factor: np.mean(contributions) 
                    for factor, contributions in aggregated_contributions.items()
                }
                factor_analysis['average_contributions'] = avg_contributions
                factor_analysis['components_with_factor_data'] = component_count
        
        except Exception as e:
            logger.error(f"Error extracting factor analysis: {e}")
        
        return factor_analysis