"""
Portfolio Data Provider for portfolio risk analysis system.
Handles loading and processing portfolio definitions from YAML files,
constructs PortfolioGraph using Spark framework builders.
"""

from typing import List, Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml

if TYPE_CHECKING:
    from spark.portfolio.graph import PortfolioGraph

logger = logging.getLogger(__name__)


class PortfolioDataProvider:
    """
    Provides comprehensive access to portfolio data through YAML-defined portfolio configurations.
    
    Handles portfolio structure definition, data loading, and PortfolioGraph construction
    using Spark framework builders.
    """
    
    def __init__(self, portfolio_yaml_path: str, strict_mode: bool = False):
        """
        Initialize portfolio data provider from YAML configuration.
        
        Args:
            portfolio_yaml_path: Path to portfolio YAML configuration file
            strict_mode: If True, fail when components are missing in data. If False, skip with warning.
        """
        self.portfolio_yaml_path = Path(portfolio_yaml_path)
        self._data: Optional[pd.DataFrame] = None
        self._portfolio_config: Optional[Dict[str, Any]] = None
        self._portfolio_graph: Optional['PortfolioGraph'] = None
        self.strict_mode = strict_mode
        
        # Frequency management for data providers
        self.frequency_manager = None
        self.current_frequency = "B"  # Default to business daily
        
        # Date range filtering
        self.date_range_start: Optional[datetime] = None
        self.date_range_end: Optional[datetime] = None
        
        self._load_config_and_build_graph()
    
    def set_frequency_manager(self, frequency_manager):
        """
        Set frequency manager to coordinate frequency with DataAccessService.
        
        Args:
            frequency_manager: FrequencyManager instance from DataAccessService
        """
        self.frequency_manager = frequency_manager
        if frequency_manager:
            self.current_frequency = frequency_manager.current_frequency
            logger.info(f"PortfolioDataProvider frequency set to: {self.current_frequency}")
    
    def set_date_range(self, start_date: Optional[datetime], end_date: Optional[datetime]):
        """
        Set date range filter and reload data with filtering applied at source.
        
        Args:
            start_date: Start date for filtering (inclusive)
            end_date: End date for filtering (inclusive)
        """
        # Check if date range actually changed
        date_range_changed = (
            self.date_range_start != start_date or 
            self.date_range_end != end_date
        )
        
        if date_range_changed:
            self.date_range_start = start_date
            self.date_range_end = end_date
            logger.info(f"PortfolioDataProvider date range set to: [{start_date}, {end_date}]")
            
            # Reload and rebuild with filtered data from source
            self._load_config_and_build_graph()
    
    def _resample_returns_series(self, series: pd.Series) -> pd.Series:
        """
        Resample returns series using compound return calculation.
        
        Args:
            series: Returns series to resample
            
        Returns:
            Resampled series or original series if no resampling needed
        """
        if not self.frequency_manager or not self.frequency_manager.is_resampled or series.empty:
            return series
        
        freq = self.frequency_manager.current_frequency
        try:
            # Use compound return calculation: (1+r).prod() - 1
            resampled = series.resample(freq).apply(lambda x: (1 + x).prod() - 1)
            resampled.name = series.name
            logger.debug(f"PortfolioDataProvider resampled {series.name} from {len(series)} to {len(resampled)} observations at {freq}")
            return resampled.dropna()
            
        except Exception as e:
            logger.error(f"Error resampling series {series.name} in PortfolioDataProvider: {e}")
            return series
    
    def _resample_returns_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample returns DataFrame using compound return calculation.
        
        Args:
            df: Returns DataFrame to resample
            
        Returns:
            Resampled DataFrame or original DataFrame if no resampling needed
        """
        if not self.frequency_manager or not self.frequency_manager.is_resampled or df.empty:
            return df
        
        freq = self.frequency_manager.current_frequency
        try:
            # Apply compound return calculation to each column
            resampled = df.resample(freq).apply(lambda x: (1 + x).prod() - 1)
            logger.debug(f"PortfolioDataProvider resampled DataFrame from {len(df)} to {len(resampled)} observations at {freq}")
            return resampled.dropna()
            
        except Exception as e:
            logger.error(f"Error resampling DataFrame in PortfolioDataProvider: {e}")
            return df
    
    def _load_config_and_build_graph(self) -> None:
        """Load YAML configuration and build PortfolioGraph."""
        try:
            if not self.portfolio_yaml_path.exists():
                raise FileNotFoundError(f"Portfolio YAML file not found: {self.portfolio_yaml_path}")
            
            # Load YAML configuration
            with open(self.portfolio_yaml_path, 'r') as f:
                self._portfolio_config = yaml.safe_load(f)
            
            # Load time series data from parquet file specified in YAML
            self._load_time_series_data()
            
            # Build PortfolioGraph using Spark framework
            self._build_portfolio_graph()
            
            logger.info(f"Loaded portfolio configuration: {self._portfolio_config.get('name', 'Unknown')}")
            logger.info(f"Built PortfolioGraph with {len(self._portfolio_graph.components) if self._portfolio_graph else 0} components")
                       
        except Exception as e:
            logger.error(f"Failed to load portfolio configuration from {self.portfolio_yaml_path}: {e}")
            raise
    
    def _load_time_series_data(self) -> None:
        """Load time series data from parquet file specified in YAML."""
        data_sources = self._portfolio_config.get('data_sources', {})
        portfolio_data_path = data_sources.get('portfolio_data')
        
        if not portfolio_data_path:
            raise ValueError("No portfolio_data specified in YAML data_sources")
        
        # Make path relative to YAML file location
        data_path = self.portfolio_yaml_path.parent / portfolio_data_path
        
        if not data_path.exists():
            raise FileNotFoundError(f"Portfolio data file not found: {data_path}")
        
        self._data = pd.read_parquet(data_path)
        
        # Validate required columns
        basic_columns = {'component_id', 'date'}
        if not basic_columns.issubset(self._data.columns):
            missing = basic_columns - set(self._data.columns)
            raise ValueError(f"Missing required columns in portfolio data: {missing}")
        
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self._data['date']):
            self._data['date'] = pd.to_datetime(self._data['date'])
        
        # Apply date range filtering at source if set
        self._apply_date_range_filter()
        
        # Sort by date and component for consistency
        self._data = self._data.sort_values(['date', 'component_id'])
        
        logger.info(f"Loaded time series data: {len(self._data)} records, "
                   f"{self._data['component_id'].nunique()} components")
    
    def _apply_date_range_filter(self):
        """Apply date range filter to loaded data if date range is set."""
        if self._data is None or self._data.empty:
            return
            
        original_count = len(self._data)
        
        # Apply date filtering if date range is set
        if self.date_range_start is not None or self.date_range_end is not None:
            if self.date_range_start is not None:
                self._data = self._data[self._data['date'] >= pd.Timestamp(self.date_range_start)]
                
            if self.date_range_end is not None:
                self._data = self._data[self._data['date'] <= pd.Timestamp(self.date_range_end)]
            
            filtered_count = len(self._data)
            logger.info(f"Applied date range filter: {original_count} -> {filtered_count} records")
            
            if filtered_count == 0:
                logger.warning("Date range filter resulted in no data. Check date range settings.")
        else:
            logger.debug("No date range filter applied - using all available data")
    
    def _validate_component_mapping(self) -> Dict[str, Any]:
        """
        Validate that all components defined in YAML exist in the data.
        
        Returns:
            Dictionary with validation results including missing components
        """
        validation_result = {
            "valid": True,
            "missing_components": [],
            "found_components": [],
            "data_components": [],
            "warnings": []
        }
        
        if self._data is None:
            validation_result["valid"] = False
            validation_result["warnings"].append("No data loaded")
            return validation_result
        
        # Get all unique component_ids from the data
        data_structure = self._portfolio_config.get('data_structure', {})
        component_id_col = data_structure.get('component_id_column', 'component_id')
        data_components = set(self._data[component_id_col].unique())
        validation_result["data_components"] = sorted(list(data_components))
        
        # Check each component defined in YAML
        components = self._portfolio_config.get('components', [])
        for comp in components:
            component_path = comp['path']
            if component_path in data_components:
                validation_result["found_components"].append(component_path)
            else:
                validation_result["missing_components"].append(component_path)
                logger.warning(f"Component '{component_path}' defined in YAML but not found in data")
        
        # Check if we have any missing components
        if validation_result["missing_components"]:
            validation_result["valid"] = False
            missing_count = len(validation_result["missing_components"])
            total_count = len(components)
            
            logger.warning(f"Missing {missing_count}/{total_count} components from data file")
            logger.warning(f"Missing components: {validation_result['missing_components'][:5]}{'...' if missing_count > 5 else ''}")
            logger.info(f"Available components in data: {sorted(list(data_components))[:10]}{'...' if len(data_components) > 10 else ''}")
            
            if self.strict_mode:
                raise ValueError(
                    f"Missing {missing_count} components in portfolio data. "
                    f"Missing: {validation_result['missing_components'][:3]}... "
                    f"Set strict_mode=False to skip missing components."
                )
        
        return validation_result
    
    def _get_latest_component_data(self, component_id: str) -> Optional[pd.DataFrame]:
        """
        Get the latest data for a component from the time series data.
        
        Args:
            component_id: Component identifier
            
        Returns:
            DataFrame with component data or None if not found
        """
        if self._data is None:
            return None
        
        data_structure = self._portfolio_config.get('data_structure', {})
        component_id_col = data_structure.get('component_id_column', 'component_id')
        
        component_data = self._data[self._data[component_id_col] == component_id]
        if component_data.empty:
            logger.debug(f"No data found for component: {component_id}")
            return None
        
        # Sort by date and return all data for this component
        date_col = data_structure.get('date_column', 'date')
        return component_data.sort_values(date_col)
    
    def _build_portfolio_graph(self) -> None:
        """Build PortfolioGraph using Spark framework builders."""
        try:
            # Validate component mapping first
            validation_result = self._validate_component_mapping()
            if not validation_result["valid"] and self.strict_mode:
                # Validation will have already raised an error in strict mode
                return
            
            # Import Spark framework classes
            from spark.portfolio.graph import PortfolioGraph
            from spark.portfolio.builders import PortfolioBuilder
            from spark.portfolio.builder_multiplicative import PortfolioBuilderMultiplicative
            from spark.portfolio.builder_sum import PortfolioBuilderSum
            
            # Get builder settings from YAML
            builder_settings = self._portfolio_config.get('builder_settings', {})
            builder_class_name = builder_settings.get('builder_class', 'PortfolioBuilder')
            
            # Select correct builder class
            builder_classes = {
                'PortfolioBuilder': PortfolioBuilder,
                'PortfolioBuilderMultiplicative': PortfolioBuilderMultiplicative,
                'PortfolioBuilderSum': PortfolioBuilderSum
            }
            
            if builder_class_name not in builder_classes:
                raise ValueError(f"Unknown builder class: {builder_class_name}")
            
            BuilderClass = builder_classes[builder_class_name]
            
            # Extract builder parameters (excluding builder_class)
            builder_params = {k: v for k, v in builder_settings.items() if k != 'builder_class'}
            
            # Create builder instance
            builder = BuilderClass(**builder_params)
            
            # Get component definitions from YAML
            components = self._portfolio_config.get('components', [])
            
            # Convert to format expected by from_paths and include weight data
            path_data = []
            skipped_components = []
            
            for comp in components:
                component_data = self._get_latest_component_data(comp['path'])
                
                # Skip components with no data if not in strict mode
                if component_data is None and not self.strict_mode:
                    skipped_components.append(comp['path'])
                    logger.debug(f"Skipping component '{comp['path']}' - no data available")
                    continue
                
                path_entry = {
                    'path': comp['path'],
                    'component_type': comp.get('component_type', 'leaf'),
                    'is_overlay': comp.get('is_overlay', False),
                    'name': comp.get('name', comp['path'])
                }
                
                # Add weight data if available
                if component_data is not None:
                    data_structure = self._portfolio_config.get('data_structure', {})
                    portfolio_weight_col = data_structure.get('portfolio_weight_column', 'portfolio_weight')
                    benchmark_weight_col = data_structure.get('benchmark_weight_column', 'benchmark_weight')
                    
                    if portfolio_weight_col in component_data.columns:
                        path_entry['portfolio_weight'] = float(component_data[portfolio_weight_col].iloc[-1])
                    
                    if benchmark_weight_col in component_data.columns:
                        path_entry['benchmark_weight'] = float(component_data[benchmark_weight_col].iloc[-1])
                
                path_data.append(path_entry)
            
            if skipped_components:
                logger.warning(f"Skipped {len(skipped_components)} components without data: {skipped_components[:5]}{'...' if len(skipped_components) > 5 else ''}")
            
            # Build portfolio from paths with weight data included
            builder.from_paths(path_data)
            
            # Assign time series data (returns) to components BEFORE building
            self._assign_time_series_data(builder)
            
            # Get the constructed PortfolioGraph (with time series data)
            self._portfolio_graph = builder.build()
            
            # Set root_id from builder_settings if specified
            root_id = builder_settings.get('root_id')
            if root_id:
                self._portfolio_graph.root_id = root_id
                logger.info(f"Set PortfolioGraph root_id to: {root_id}")
            else:
                logger.warning("No root_id specified in builder_settings")
            
            logger.info(f"Built PortfolioGraph using {builder_class_name} with time series data")
            
        except ImportError as e:
            logger.error(f"Failed to import Spark framework classes: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to build PortfolioGraph: {e}")
            raise
    
    def _assign_time_series_data(self, builder) -> None:
        """Assign time series data from DataFrame to portfolio components using builder."""
        try:
            # Get data structure mappings from YAML config
            data_structure = self._portfolio_config.get('data_structure', {})
            component_id_col = data_structure.get('component_id_column', 'component_id')
            date_col = data_structure.get('date_column', 'date')
            portfolio_weight_col = data_structure.get('portfolio_weight_column', 'portfolio_weight')
            benchmark_weight_col = data_structure.get('benchmark_weight_column', 'benchmark_weight')
            portfolio_return_col = data_structure.get('portfolio_return_column', 'portfolio_return')
            benchmark_return_col = data_structure.get('benchmark_return_column', 'benchmark_return')
            
            # Group data by component
            component_groups = self._data.groupby(component_id_col)
            
            logger.info(f"Assigning time series data for {len(component_groups)} components")
            
            for component_id, component_data in component_groups:
                # Sort by date
                component_data = component_data.sort_values(date_col)
                
                # Prepare data dict for this component
                data_dict = {}
                
                # Add time series returns only (weights are now handled by from_paths)
                
                if portfolio_return_col in component_data.columns:
                    # Keep returns as time series
                    data_dict['portfolio_return'] = component_data.set_index(date_col)[portfolio_return_col]
                
                if benchmark_return_col in component_data.columns:
                    # Keep returns as time series
                    data_dict['benchmark_return'] = component_data.set_index(date_col)[benchmark_return_col]
                
                # Use builder's add_data method to assign time series data
                if data_dict:
                    try:
                        builder.add_data(component_id, data_dict)
                        logger.debug(f"Assigned time series data for component: {component_id}")
                    except Exception as e:
                        if self.strict_mode:
                            raise
                        logger.warning(f"Failed to assign data for component '{component_id}': {e}")
            
            logger.info("Time series data assignment completed")
            
        except Exception as e:
            logger.error(f"Failed to assign time series data: {e}")
            raise
    
    def get_portfolio_graph(self) -> Optional['PortfolioGraph']:
        """
        Get the constructed PortfolioGraph.
        
        Returns:
            PortfolioGraph instance or None if not built
        """
        return self._portfolio_graph
    
    def get_portfolio_metadata(self) -> Dict[str, Any]:
        """
        Get portfolio metadata from YAML configuration.
        
        Returns:
            Dictionary with portfolio metadata
        """
        if not self._portfolio_config:
            return {}
        
        return {
            'name': self._portfolio_config.get('name'),
            'description': self._portfolio_config.get('description'),
            'builder_settings': self._portfolio_config.get('builder_settings', {}),
            'data_structure': self._portfolio_config.get('data_structure', {})
        }
    
    def load_portfolio_data(self) -> pd.DataFrame:
        """
        Load all portfolio time series data.
        
        Returns:
            DataFrame with all portfolio data
        """
        if self._data is None:
            raise RuntimeError("Portfolio data not loaded")
        
        return self._data.copy()
    
    def get_component_returns(self, component_id: str, return_type: str) -> pd.Series:
        """
        Get return time series for a component using PortfolioGraph methods.
        
        Args:
            component_id: Component identifier
            return_type: Type of returns ('portfolio', 'benchmark', or 'active')
            
        Returns:
            Series with dates as index and returns as values
        """
        if self._portfolio_graph is None:
            raise RuntimeError("PortfolioGraph not built")
        
        if return_type not in ['portfolio', 'benchmark', 'active']:
            raise ValueError(f"Invalid return_type: {return_type}. Must be 'portfolio', 'benchmark', or 'active'")
        
        try:
            # Use PortfolioGraph methods for return calculation
            if return_type == 'portfolio':
                metric = self._portfolio_graph.portfolio_returns(component_id)
            elif return_type == 'benchmark':
                metric = self._portfolio_graph.benchmark_returns(component_id)
            elif return_type == 'active':
                metric = self._portfolio_graph.excess_returns(component_id)
            
            if metric is None:
                logger.warning(f"No {return_type} returns found for component: {component_id}")
                return pd.Series(dtype=float, name=f"{component_id}_{return_type}")
            
            # Convert metric to pandas Series
            metric_value = metric.value()
            
            if isinstance(metric_value, pd.Series):
                result_series = metric_value.copy()
                result_series.name = f"{component_id}_{return_type}"
                logger.debug(f"Created {return_type} returns series for {component_id}: {len(result_series)} observations")
                # Apply resampling if frequency manager is set
                return self._resample_returns_series(result_series)
            elif isinstance(metric_value, (int, float)):
                # Scalar metric - create single-value series
                result_series = pd.Series([metric_value], name=f"{component_id}_{return_type}")
                logger.debug(f"Created {return_type} returns scalar for {component_id}: {metric_value}")
                return result_series
            else:
                logger.warning(f"Unexpected metric type for {component_id}: {type(metric_value)}")
                return pd.Series(dtype=float, name=f"{component_id}_{return_type}")
                
        except Exception as e:
            logger.error(f"Failed to get {return_type} returns for component {component_id}: {e}")
            return pd.Series(dtype=float, name=f"{component_id}_{return_type}")
    
    def get_component_weights(self, component_id: str, weight_type: str) -> pd.Series:
        """
        Get weight values for a component using PortfolioGraph methods.
        
        Args:
            component_id: Component identifier
            weight_type: Type of weights ('portfolio', 'benchmark', or 'active')
            
        Returns:
            Series with component weight values (converted from Dict for backward compatibility)
        """
        if self._portfolio_graph is None:
            raise RuntimeError("PortfolioGraph not built")
        
        if weight_type not in ['portfolio', 'benchmark', 'active']:
            raise ValueError(f"Invalid weight_type: {weight_type}. Must be 'portfolio', 'benchmark', or 'active'")
        
        try:
            # Use PortfolioGraph methods for weight calculation
            if weight_type == 'portfolio':
                weights_dict = self._portfolio_graph.portfolio_weights(component_id)
            elif weight_type == 'benchmark':
                weights_dict = self._portfolio_graph.benchmark_weights(component_id)
            elif weight_type == 'active':
                # Get both portfolio and benchmark weights, then calculate active
                portfolio_weights = self._portfolio_graph.portfolio_weights(component_id)
                benchmark_weights = self._portfolio_graph.benchmark_weights(component_id)
                
                # Calculate active weights (portfolio - benchmark)
                weights_dict = {}
                all_components = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
                for comp_id in all_components:
                    port_weight = portfolio_weights.get(comp_id, 0.0)
                    bench_weight = benchmark_weights.get(comp_id, 0.0)
                    weights_dict[comp_id] = port_weight - bench_weight
            
            if not weights_dict:
                logger.warning(f"No {weight_type} weights found for component: {component_id}")
                return pd.Series(dtype=float, name=f"{component_id}_{weight_type}_weight")
            
            # Convert dict to Series for backward compatibility
            result_series = pd.Series(weights_dict, name=f"{component_id}_{weight_type}_weight")
            result_series = result_series.sort_index()  # Sort by component IDs
            
            logger.debug(f"Created {weight_type} weights series for {component_id}: {len(result_series)} components")
            return result_series
                
        except Exception as e:
            logger.error(f"Failed to get {weight_type} weights for component {component_id}: {e}")
            return pd.Series(dtype=float, name=f"{component_id}_{weight_type}_weight")
    
    def get_all_component_ids(self) -> List[str]:
        """
        Get list of all component IDs from PortfolioGraph.
        
        Returns:
            List of component IDs
        """
        if self._portfolio_graph is None:
            raise RuntimeError("PortfolioGraph not built")
        
        components = sorted(self._portfolio_graph.components.keys())
        logger.debug(f"Found {len(components)} components")
        return components
    
    def get_leaf_components(self) -> List[str]:
        """
        Get list of leaf components (components without children) from PortfolioGraph.
        
        Returns:
            List of leaf component IDs
        """
        if self._portfolio_graph is None:
            raise RuntimeError("PortfolioGraph not built")
        
        leaf_components = []
        for comp_id, component in self._portfolio_graph.components.items():
            # Use component's is_leaf() method to check if it's a leaf
            if component.is_leaf():
                leaf_components.append(comp_id)
        
        leaf_components = sorted(leaf_components)
        logger.debug(f"Found {len(leaf_components)} leaf components")
        return leaf_components
    
    def get_node_components(self) -> List[str]:
        """
        Get list of node components (components with children) from PortfolioGraph.
        
        Returns:
            List of node component IDs
        """
        if self._portfolio_graph is None:
            raise RuntimeError("PortfolioGraph not built")
        
        node_components = []
        for comp_id, component in self._portfolio_graph.components.items():
            # Use component's is_leaf() method - nodes are non-leaf components
            if not component.is_leaf():
                node_components.append(comp_id)
        
        node_components = sorted(node_components)
        logger.debug(f"Found {len(node_components)} node components")
        return node_components
    
    def get_component_hierarchy(self) -> Dict[str, List[str]]:
        """
        Get component hierarchy mapping from PortfolioGraph.
        
        Returns:
            Dictionary mapping parent component IDs to lists of child component IDs
        """
        if self._portfolio_graph is None:
            raise RuntimeError("PortfolioGraph not built")
        
        return self._portfolio_graph.adjacency_list.copy()
    
    def get_returns_matrix(self, return_type: str) -> pd.DataFrame:
        """
        Get returns matrix with all components as columns using PortfolioGraph methods.
        
        Args:
            return_type: Type of returns ('portfolio', 'benchmark', or 'active')
            
        Returns:
            DataFrame with dates as index and components as columns
        """
        if self._portfolio_graph is None:
            raise RuntimeError("PortfolioGraph not built")
        
        if return_type not in ['portfolio', 'benchmark', 'active']:
            raise ValueError(f"Invalid return_type: {return_type}. Must be 'portfolio', 'benchmark', or 'active'")
        
        try:
            # Use collect_metrics to gather returns from all leaf components
            if return_type == 'active':
                metric_name = 'excess_return'  # Use appropriate metric name for active returns
            else:
                metric_name = f'{return_type}_return'
            
            # Get root component ID (assuming single root or first available)
            root_id = self._portfolio_graph.root_id
            if root_id is None:
                # Fallback: get first available component as root
                root_candidates = [comp_id for comp_id, comp in self._portfolio_graph.components.items() 
                                 if not comp.parent_ids]
                root_id = root_candidates[0] if root_candidates else list(self._portfolio_graph.components.keys())[0]
            
            # Collect metrics from all leaf components
            metrics_dict = self._portfolio_graph.collect_metrics(root_id, metric_name, include_nodes=False)
            
            if not metrics_dict:
                logger.warning(f"No {return_type} returns found in portfolio graph")
                return pd.DataFrame()
            
            # Convert collected metrics to DataFrame
            # Handle both scalar and series metrics
            series_data = {}
            for comp_id, metric_value in metrics_dict.items():
                if isinstance(metric_value, pd.Series):
                    series_data[comp_id] = metric_value
                elif isinstance(metric_value, (int, float)):
                    # Create single-value series for scalar metrics
                    series_data[comp_id] = pd.Series([metric_value])
            
            if not series_data:
                logger.warning(f"No valid time series data found for {return_type} returns")
                return pd.DataFrame()
            
            # Create DataFrame from series data
            returns_matrix = pd.DataFrame(series_data)
            
            # Fill NaN values with 0 for consistency
            returns_matrix = returns_matrix.fillna(0.0)
            
            # Sort columns for consistent output
            returns_matrix = returns_matrix.reindex(sorted(returns_matrix.columns), axis=1)
            
            logger.debug(f"Created {return_type} returns matrix using PortfolioGraph: {returns_matrix.shape[0]} dates, "
                        f"{returns_matrix.shape[1]} components")
            # Apply resampling if frequency manager is set
            return self._resample_returns_dataframe(returns_matrix)
            
        except Exception as e:
            logger.error(f"Failed to create {return_type} returns matrix: {e}")
            return pd.DataFrame()
    
    def get_weights_snapshot(self, date: datetime = None) -> pd.DataFrame:
        """
        Get cross-section of weights using PortfolioGraph methods.
        
        Args:
            date: Date for the snapshot (currently ignored as PortfolioGraph manages time-based logic)
            
        Returns:
            DataFrame with components and their current weights
        """
        if self._portfolio_graph is None:
            raise RuntimeError("PortfolioGraph not built")
        
        try:
            # Get root component ID
            root_id = self._portfolio_graph.root_id
            if root_id is None:
                # Fallback: get first available component as root
                root_candidates = [comp_id for comp_id, comp in self._portfolio_graph.components.items() 
                                 if not comp.parent_ids]
                root_id = root_candidates[0] if root_candidates else list(self._portfolio_graph.components.keys())[0]
            
            # Get portfolio and benchmark weights using PortfolioGraph methods
            portfolio_weights = self._portfolio_graph.portfolio_weights(root_id)
            benchmark_weights = self._portfolio_graph.benchmark_weights(root_id)
            
            # Combine all components from both weight dictionaries
            all_components = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
            
            if not all_components:
                logger.warning("No weight data found in portfolio graph")
                return pd.DataFrame()
            
            # Create snapshot data
            snapshot_data = []
            for component_id in sorted(all_components):
                row = {
                    'component_id': component_id,
                    'portfolio_weight': portfolio_weights.get(component_id, 0.0),
                    'benchmark_weight': benchmark_weights.get(component_id, 0.0)
                }
                # Calculate active weight
                row['active_weight'] = row['portfolio_weight'] - row['benchmark_weight']
                snapshot_data.append(row)
            
            # Create DataFrame
            snapshot = pd.DataFrame(snapshot_data)
            
            # Add date information if provided
            if date is not None:
                snapshot['date'] = pd.Timestamp(date)
                logger.debug(f"Created weights snapshot for requested date {date}: {len(snapshot)} components")
            else:
                logger.debug(f"Created current weights snapshot: {len(snapshot)} components")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to create weights snapshot: {e}")
            return pd.DataFrame()
    
    def get_component_children(self, parent_id: str) -> List[str]:
        """
        Get direct children of a component from PortfolioGraph.
        
        Args:
            parent_id: Parent component ID
            
        Returns:
            List of child component IDs
        """
        if self._portfolio_graph is None:
            raise RuntimeError("PortfolioGraph not built")
        
        children = self._portfolio_graph.adjacency_list.get(parent_id, [])
        logger.debug(f"Component {parent_id} has {len(children)} children")
        return children
    
    def get_component_parent(self, component_id: str) -> Optional[str]:
        """
        Get parent of a component from PortfolioGraph.
        
        Args:
            component_id: Component ID
            
        Returns:
            Parent component ID or None if no parent
        """
        if self._portfolio_graph is None:
            raise RuntimeError("PortfolioGraph not built")
        
        for parent, children in self._portfolio_graph.adjacency_list.items():
            if component_id in children:
                return parent
        
        return None
    
    def get_component_metadata(self, component_id: str) -> Dict[str, Any]:
        """
        Get component metadata including overlay status and other configuration.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Dictionary with component metadata including:
            - is_overlay: Whether component is an overlay strategy
            - component_type: 'leaf' or 'node' 
            - name: Display name
        """
        try:
            # Get component from portfolio graph
            if self._portfolio_graph and component_id in self._portfolio_graph.components:
                component = self._portfolio_graph.components[component_id]
                return {
                    "is_overlay": getattr(component, 'is_overlay', False),
                    "component_type": "leaf" if len(self.get_component_children(component_id)) == 0 else "node",
                    "name": component_id,
                    "path": component_id
                }
            
            # Fallback: check configuration data
            if self._portfolio_config and 'components' in self._portfolio_config:
                for comp in self._portfolio_config['components']:
                    if comp.get('path') == component_id:
                        return {
                            "is_overlay": comp.get('is_overlay', False),
                            "component_type": comp.get('component_type', 'leaf'),
                            "name": comp.get('name', component_id),
                            "path": comp['path']
                        }
            
            # Default metadata if not found
            return {
                "is_overlay": False,
                "component_type": "leaf" if len(self.get_component_children(component_id)) == 0 else "node",
                "name": component_id,
                "path": component_id
            }
            
        except Exception as e:
            logger.error(f"Error getting metadata for {component_id}: {e}")
            return {
                "is_overlay": False,
                "component_type": "leaf",
                "name": component_id,
                "path": component_id
            }
    
    def validate_data_consistency(self) -> Dict[str, any]:
        """
        Validate data consistency and completeness.
        
        Returns:
            Dictionary with validation results
        """
        if self._data is None or self._portfolio_graph is None:
            return {"valid": False, "error": "Portfolio data or graph not loaded"}
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "total_records": len(self._data),
            "unique_components": self._data['component_id'].nunique(),
            "unique_dates": self._data['date'].nunique(),
            "date_range": (self._data['date'].min(), self._data['date'].max()),
            "portfolio_graph_components": len(self._portfolio_graph.components)
        }
        
        # Check for missing data in key columns
        for column in ['portfolio_return', 'benchmark_return', 'portfolio_weight', 'benchmark_weight']:
            if column in self._data.columns:
                missing_count = self._data[column].isna().sum()
                if missing_count > 0:
                    validation_result['warnings'].append(f"{missing_count} missing values in {column}")
        
        # Check hierarchy consistency with PortfolioGraph
        hierarchy = self.get_component_hierarchy()
        if hierarchy:
            validation_result['hierarchy_nodes'] = len(hierarchy)
            validation_result['hierarchy_depth'] = self._calculate_hierarchy_depth()
        
        # Check consistency between time series data and portfolio graph
        data_components = set(self._data['component_id'].unique())
        graph_components = set(self._portfolio_graph.components.keys())
        missing_in_data = graph_components - data_components
        missing_in_graph = data_components - graph_components
        
        if missing_in_data:
            validation_result['warnings'].append(f"{len(missing_in_data)} components in graph but not in data: {list(missing_in_data)[:5]}")
        if missing_in_graph:
            validation_result['warnings'].append(f"{len(missing_in_graph)} components in data but not in graph: {list(missing_in_graph)[:5]}")
        
        validation_result['valid'] = len(validation_result['errors']) == 0
        
        logger.info(f"Portfolio data validation: {'PASS' if validation_result['valid'] else 'FAIL'}")
        return validation_result
    
    def _calculate_hierarchy_depth(self) -> int:
        """Calculate maximum hierarchy depth from PortfolioGraph."""
        if self._portfolio_graph is None:
            return 0
        
        hierarchy = self.get_component_hierarchy()
        if not hierarchy:
            return 0
        
        max_depth = 0
        
        def get_depth(component_id: str, current_depth: int = 0) -> int:
            children = hierarchy.get(component_id, [])
            if not children:
                return current_depth
            
            max_child_depth = current_depth
            for child in children:
                child_depth = get_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        
        for root_component in hierarchy.keys():
            depth = get_depth(root_component)
            max_depth = max(max_depth, depth)
        
        return max_depth