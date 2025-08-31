"""
Portfolio Data Provider for portfolio risk analysis system.
Handles loading and processing portfolio component data from parquet files.
"""

from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PortfolioDataProvider:
    """
    Provides comprehensive access to portfolio component data from parquet files.
    
    Handles portfolio data with component returns, weights, and hierarchy information.
    """
    
    def __init__(self, portfolio_data_path: str):
        """
        Initialize portfolio data provider.
        
        Args:
            portfolio_data_path: Path to portfolio data parquet file
        """
        self.portfolio_data_path = Path(portfolio_data_path)
        self._data: Optional[pd.DataFrame] = None
        self._component_hierarchy: Optional[Dict[str, List[str]]] = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load portfolio data from parquet file."""
        try:
            if not self.portfolio_data_path.exists():
                raise FileNotFoundError(f"Portfolio data file not found: {self.portfolio_data_path}")
            
            self._data = pd.read_parquet(self.portfolio_data_path)
            
            # Validate required columns for basic functionality
            basic_columns = {'component_id', 'date'}
            if not basic_columns.issubset(self._data.columns):
                missing = basic_columns - set(self._data.columns)
                raise ValueError(f"Missing required columns in portfolio data: {missing}")
            
            # Convert date column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(self._data['date']):
                self._data['date'] = pd.to_datetime(self._data['date'])
            
            # Sort by date and component for consistency
            self._data = self._data.sort_values(['date', 'component_id'])
            
            # Build component hierarchy from component IDs (assuming path-based IDs)
            self._build_hierarchy()
            
            logger.info(f"Loaded portfolio data: {len(self._data)} records, "
                       f"{self._data['component_id'].nunique()} components")
                       
        except Exception as e:
            logger.error(f"Failed to load portfolio data from {self.portfolio_data_path}: {e}")
            raise
    
    def _build_hierarchy(self) -> None:
        """Build component hierarchy from component IDs."""
        hierarchy = {}
        all_components = set(self._data['component_id'].unique())
        
        # For each component, find its children based on path structure
        for component in all_components:
            children = []
            component_path = component
            
            for other_component in all_components:
                if other_component != component:
                    # Check if other_component is a direct child
                    if other_component.startswith(component_path + "/"):
                        # Ensure it's a direct child, not a grandchild
                        remaining_path = other_component[len(component_path + "/"):]
                        if "/" not in remaining_path:
                            children.append(other_component)
            
            if children:
                hierarchy[component] = sorted(children)
        
        self._component_hierarchy = hierarchy
        logger.debug(f"Built component hierarchy with {len(hierarchy)} parent nodes")
    
    def load_portfolio_data(self) -> pd.DataFrame:
        """
        Load all portfolio data.
        
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
        Get list of all component IDs.
        
        Returns:
            List of component IDs
        """
        if self._data is None:
            raise RuntimeError("Portfolio data not loaded")
        
        components = sorted(self._data['component_id'].unique())
        logger.debug(f"Found {len(components)} components")
        return components
    
    def get_leaf_components(self) -> List[str]:
        """
        Get list of leaf components (components without children).
        
        Returns:
            List of leaf component IDs
        """
        all_components = set(self.get_all_component_ids())
        parent_components = set(self._component_hierarchy.keys()) if self._component_hierarchy else set()
        
        leaf_components = sorted(all_components - parent_components)
        logger.debug(f"Found {len(leaf_components)} leaf components")
        return leaf_components
    
    def get_node_components(self) -> List[str]:
        """
        Get list of node components (components with children).
        
        Returns:
            List of node component IDs
        """
        if not self._component_hierarchy:
            return []
        
        node_components = sorted(self._component_hierarchy.keys())
        logger.debug(f"Found {len(node_components)} node components")
        return node_components
    
    def get_component_hierarchy(self) -> Dict[str, List[str]]:
        """
        Get component hierarchy mapping.
        
        Returns:
            Dictionary mapping parent component IDs to lists of child component IDs
        """
        if not self._component_hierarchy:
            return {}
        
        return self._component_hierarchy.copy()
    
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
        Get direct children of a component.
        
        Args:
            parent_id: Parent component ID
            
        Returns:
            List of child component IDs
        """
        if not self._component_hierarchy:
            return []
        
        children = self._component_hierarchy.get(parent_id, [])
        logger.debug(f"Component {parent_id} has {len(children)} children")
        return children
    
    def get_component_parent(self, component_id: str) -> Optional[str]:
        """
        Get parent of a component.
        
        Args:
            component_id: Component ID
            
        Returns:
            Parent component ID or None if no parent
        """
        if not self._component_hierarchy:
            return None
        
        for parent, children in self._component_hierarchy.items():
            if component_id in children:
                return parent
        
        return None
    
    def validate_data_consistency(self) -> Dict[str, any]:
        """
        Validate data consistency and completeness.
        
        Returns:
            Dictionary with validation results
        """
        if self._data is None:
            return {"valid": False, "error": "No data loaded"}
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "total_records": len(self._data),
            "unique_components": self._data['component_id'].nunique(),
            "unique_dates": self._data['date'].nunique(),
            "date_range": (self._data['date'].min(), self._data['date'].max())
        }
        
        # Check for missing data in key columns
        for column in ['portfolio_return', 'benchmark_return', 'portfolio_weight', 'benchmark_weight']:
            if column in self._data.columns:
                missing_count = self._data[column].isna().sum()
                if missing_count > 0:
                    validation_result['warnings'].append(f"{missing_count} missing values in {column}")
        
        # Check hierarchy consistency
        if self._component_hierarchy:
            validation_result['hierarchy_nodes'] = len(self._component_hierarchy)
            validation_result['hierarchy_depth'] = self._calculate_hierarchy_depth()
        
        validation_result['valid'] = len(validation_result['errors']) == 0
        
        logger.info(f"Portfolio data validation: {'PASS' if validation_result['valid'] else 'FAIL'}")
        return validation_result
    
    def _calculate_hierarchy_depth(self) -> int:
        """Calculate maximum hierarchy depth."""
        if not self._component_hierarchy:
            return 0
        
        max_depth = 0
        
        def get_depth(component_id: str, current_depth: int = 0) -> int:
            children = self._component_hierarchy.get(component_id, [])
            if not children:
                return current_depth
            
            max_child_depth = current_depth
            for child in children:
                child_depth = get_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        
        for root_component in self._component_hierarchy.keys():
            depth = get_depth(root_component)
            max_depth = max(max_depth, depth)
        
        return max_depth