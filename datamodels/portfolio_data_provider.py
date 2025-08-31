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
    
    def __init__(self, portfolio_yaml_path: str):
        """
        Initialize portfolio data provider from YAML configuration.
        
        Args:
            portfolio_yaml_path: Path to portfolio YAML configuration file
        """
        self.portfolio_yaml_path = Path(portfolio_yaml_path)
        self._data: Optional[pd.DataFrame] = None
        self._portfolio_config: Optional[Dict[str, Any]] = None
        self._portfolio_graph: Optional['PortfolioGraph'] = None
        self._load_config_and_build_graph()
    
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
        
        # Sort by date and component for consistency
        self._data = self._data.sort_values(['date', 'component_id'])
        
        logger.info(f"Loaded time series data: {len(self._data)} records, "
                   f"{self._data['component_id'].nunique()} components")
    
    def _build_portfolio_graph(self) -> None:
        """Build PortfolioGraph using Spark framework builders."""
        try:
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
            
            # Convert to format expected by from_paths
            path_data = []
            for comp in components:
                path_data.append({
                    'path': comp['path'],
                    'component_type': comp.get('component_type', 'leaf'),
                    'is_overlay': comp.get('is_overlay', False),
                    'name': comp.get('name', comp['path'])
                })
            
            # Build portfolio from paths
            builder.from_paths(path_data)
            
            # Assign time series data to components BEFORE building
            self._assign_time_series_data(builder)
            
            # Get the constructed PortfolioGraph (with time series data)
            self._portfolio_graph = builder.build()
            
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
                
                # Add scalar weights (latest values) and time series returns
                if portfolio_weight_col in component_data.columns:
                    # Use latest weight as scalar value
                    data_dict['portfolio_weight'] = float(component_data[portfolio_weight_col].iloc[-1])
                
                if benchmark_weight_col in component_data.columns:
                    # Use latest weight as scalar value
                    data_dict['benchmark_weight'] = float(component_data[benchmark_weight_col].iloc[-1])
                
                if portfolio_return_col in component_data.columns:
                    # Keep returns as time series
                    data_dict['portfolio_return'] = component_data.set_index(date_col)[portfolio_return_col]
                
                if benchmark_return_col in component_data.columns:
                    # Keep returns as time series
                    data_dict['benchmark_return'] = component_data.set_index(date_col)[benchmark_return_col]
                
                # Use builder's add_data method to assign time series data
                if data_dict:
                    builder.add_data(component_id, data_dict)
                    logger.debug(f"Assigned time series data for component: {component_id}")
            
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
        Get return time series for a component.
        
        Args:
            component_id: Component identifier
            return_type: Type of returns ('portfolio' or 'benchmark')
            
        Returns:
            Series with dates as index and returns as values
        """
        if self._data is None:
            raise RuntimeError("Portfolio data not loaded")
        
        # Map return type to column name
        return_column_map = {
            'portfolio': 'portfolio_return',
            'benchmark': 'benchmark_return'
        }
        
        if return_type not in return_column_map:
            raise ValueError(f"Invalid return_type: {return_type}. Must be 'portfolio' or 'benchmark'")
        
        return_column = return_column_map[return_type]
        
        if return_column not in self._data.columns:
            logger.warning(f"Column {return_column} not found in data")
            return pd.Series(dtype=float, name=f"{component_id}_{return_type}")
        
        component_data = self._data[self._data['component_id'] == component_id].copy()
        
        if component_data.empty:
            logger.warning(f"No data found for component: {component_id}")
            return pd.Series(dtype=float, name=f"{component_id}_{return_type}")
        
        # Create time series
        series = component_data.set_index('date')[return_column].sort_index()
        series.name = f"{component_id}_{return_type}"
        
        logger.debug(f"Created {return_type} returns series for {component_id}: {len(series)} observations")
        return series
    
    def get_component_weights(self, component_id: str, weight_type: str) -> pd.Series:
        """
        Get weight time series for a component.
        
        Args:
            component_id: Component identifier
            weight_type: Type of weights ('portfolio' or 'benchmark')
            
        Returns:
            Series with dates as index and weights as values
        """
        if self._data is None:
            raise RuntimeError("Portfolio data not loaded")
        
        # Map weight type to column name
        weight_column_map = {
            'portfolio': 'portfolio_weight',
            'benchmark': 'benchmark_weight'
        }
        
        if weight_type not in weight_column_map:
            raise ValueError(f"Invalid weight_type: {weight_type}. Must be 'portfolio' or 'benchmark'")
        
        weight_column = weight_column_map[weight_type]
        
        if weight_column not in self._data.columns:
            logger.warning(f"Column {weight_column} not found in data")
            return pd.Series(dtype=float, name=f"{component_id}_{weight_type}_weight")
        
        component_data = self._data[self._data['component_id'] == component_id].copy()
        
        if component_data.empty:
            logger.warning(f"No data found for component: {component_id}")
            return pd.Series(dtype=float, name=f"{component_id}_{weight_type}_weight")
        
        # Create time series
        series = component_data.set_index('date')[weight_column].sort_index()
        series.name = f"{component_id}_{weight_type}_weight"
        
        logger.debug(f"Created {weight_type} weights series for {component_id}: {len(series)} observations")
        return series
    
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
            # Check if component has no children in adjacency list
            if comp_id not in self._portfolio_graph.adjacency_list or not self._portfolio_graph.adjacency_list[comp_id]:
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
        for comp_id in self._portfolio_graph.adjacency_list:
            if self._portfolio_graph.adjacency_list[comp_id]:
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
        Get returns matrix with all components as columns.
        
        Args:
            return_type: Type of returns ('portfolio' or 'benchmark')
            
        Returns:
            DataFrame with dates as index and components as columns
        """
        if self._data is None:
            raise RuntimeError("Portfolio data not loaded")
        
        return_column_map = {
            'portfolio': 'portfolio_return',
            'benchmark': 'benchmark_return'
        }
        
        if return_type not in return_column_map:
            raise ValueError(f"Invalid return_type: {return_type}. Must be 'portfolio' or 'benchmark'")
        
        return_column = return_column_map[return_type]
        
        if return_column not in self._data.columns:
            logger.warning(f"Column {return_column} not found in data")
            return pd.DataFrame()
        
        # Pivot to create returns matrix
        returns_matrix = self._data.pivot_table(
            index='date',
            columns='component_id',
            values=return_column,
            aggfunc='first'
        )
        
        # Fill NaN values with 0
        returns_matrix = returns_matrix.fillna(0.0)
        
        logger.debug(f"Created {return_type} returns matrix: {returns_matrix.shape[0]} dates, "
                    f"{returns_matrix.shape[1]} components")
        return returns_matrix
    
    def get_weights_snapshot(self, date: datetime) -> pd.DataFrame:
        """
        Get cross-section of weights at a specific date.
        
        Args:
            date: Date for the snapshot
            
        Returns:
            DataFrame with components and their weights at the specified date
        """
        if self._data is None:
            raise RuntimeError("Portfolio data not loaded")
        
        # Find data for the closest available date
        available_dates = self._data['date'].unique()
        closest_date = min(available_dates, key=lambda x: abs((x - pd.Timestamp(date)).days))
        
        snapshot_data = self._data[self._data['date'] == closest_date].copy()
        
        if snapshot_data.empty:
            logger.warning(f"No data found for date: {date}")
            return pd.DataFrame()
        
        # Select relevant columns
        weight_columns = ['component_id']
        if 'portfolio_weight' in self._data.columns:
            weight_columns.append('portfolio_weight')
        if 'benchmark_weight' in self._data.columns:
            weight_columns.append('benchmark_weight')
        
        snapshot = snapshot_data[weight_columns].copy()
        snapshot['date'] = closest_date
        
        logger.debug(f"Created weights snapshot for {date} (using {closest_date}): "
                    f"{len(snapshot)} components")
        return snapshot
    
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