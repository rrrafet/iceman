"""
PortfolioBuilderSum - Simplified Portfolio Builder using Sum Aggregation
====================================================================

A dramatically simplified portfolio builder that uses WeightPathAggregatorSum
for natural additive weight aggregation, eliminating complex normalization logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import PortfolioGraph
    from .metrics import MetricStore

from .metrics import InMemoryMetricStore


class PortfolioBuilderSum:
    """
    Simplified portfolio builder using sum-based weight aggregation.
    
    Key Features:
    - Stores absolute weights directly (no normalization)
    - Natural support for leverage, shorts, and overlays  
    - Parent weights = sum of descendant leaf weights
    - Much simpler logic (~300 lines vs 1,000+)
    """
    
    def __init__(self,
                 *,
                 root_id: str = 'portfolio',
                 allow_shorts: bool = False,
                 include_overlays_in_aggregation: bool = False,
                 normalize: bool = False,
                 delimiter: str = "/",
                 metric_store: Optional['MetricStore'] = None):
        """
        Initialize the simplified portfolio builder.
        
        Parameters
        ----------
        root_id : str, default 'portfolio'
            ID for the root component
        allow_shorts : bool, default False
            Whether to allow negative weights
        include_overlays_in_aggregation : bool, default False
            Whether to include overlay positions in parent weight aggregation
        normalize : bool, default False
            Whether to normalize the portfolio so root weight sums to 1.0.
            When True, all weights are scaled proportionally so that the root
            (sum of all non-overlay leaves) equals 1.0
        delimiter : str, default "/"
            Path delimiter character for hierarchical paths
        metric_store : MetricStore, optional
            Metric store to use. If None, creates new InMemoryMetricStore
        """
        # Configuration
        self.root_id = root_id
        self.allow_shorts = allow_shorts
        self.include_overlays_in_aggregation = include_overlays_in_aggregation
        self.normalize = normalize
        self.delimiter = delimiter
        self.metric_store = metric_store or InMemoryMetricStore()
        
        # Internal storage for component information
        self._components_info = {}  # component_id -> component data
        self._edges = []  # List of (parent_id, child_id) tuples
        
    def add_path(self, 
                 path: str, 
                 portfolio_weight: Optional[float] = None,
                 benchmark_weight: Optional[float] = None,
                 component_type: Literal['auto', 'node', 'leaf'] = 'auto',
                 is_overlay: bool = False,
                 name: Optional[str] = None,
                 data: Optional[Dict[str, Any]] = None) -> 'PortfolioBuilderSum':
        """
        Add a component to the portfolio hierarchy using path notation.
        
        Parameters
        ----------
        path : str
            Hierarchical path (e.g., "equity/us/large_cap/tech/AAPL")
        portfolio_weight : float, optional
            Absolute portfolio weight (stored as-is, no normalization)
        benchmark_weight : float, optional  
            Absolute benchmark weight (stored as-is, no normalization)
        component_type : {'auto', 'node', 'leaf'}, default 'auto'
            Component type. 'auto' infers from whether it has children
        is_overlay : bool, default False
            Whether this is an overlay strategy position
        name : str, optional
            Display name for the component
        data : dict, optional
            Additional metadata for the component
            
        Returns
        -------
        PortfolioBuilderSum
            Self for method chaining
        """
        # Validate inputs
        if not self.allow_shorts:
            if portfolio_weight is not None and portfolio_weight < 0:
                raise ValueError(f"Negative portfolio weight {portfolio_weight} not allowed for {path}")
            if benchmark_weight is not None and benchmark_weight < 0:
                raise ValueError(f"Negative benchmark weight {benchmark_weight} not allowed for {path}")
        
        # Parse path parts
        path_parts = path.split(self.delimiter)
        component_id = path
        
        # Store component information
        self._components_info[component_id] = {
            'id': component_id,
            'name': name or path_parts[-1],  # Use last part of path as default name
            'path': path,
            'path_parts': path_parts,
            'portfolio_weight': portfolio_weight,
            'benchmark_weight': benchmark_weight,
            'component_type': component_type,
            'is_overlay': is_overlay,
            'data': data or {}
        }
        
        # Create hierarchical edges automatically
        self._create_hierarchy_edges(path_parts)
        
        return self
    
    def _create_hierarchy_edges(self, path_parts: List[str]) -> None:
        """Create parent-child edges for the hierarchical path."""
        # Create edges between each level of the hierarchy
        for i in range(1, len(path_parts)):
            parent_path = self.delimiter.join(path_parts[:i])
            child_path = self.delimiter.join(path_parts[:i+1])
            
            edge = (parent_path, child_path)
            if edge not in self._edges:
                self._edges.append(edge)
                
                # Create parent component if it doesn't exist
                if parent_path not in self._components_info:
                    self._components_info[parent_path] = {
                        'id': parent_path,
                        'name': path_parts[i-1],
                        'path': parent_path,
                        'path_parts': path_parts[:i],
                        'portfolio_weight': None,  # Will be calculated by aggregation
                        'benchmark_weight': None,  # Will be calculated by aggregation
                        'component_type': 'node',
                        'is_overlay': False,
                        'data': {}
                    }
    
    def clear(self) -> 'PortfolioBuilderSum':
        """
        Clear all accumulated component and edge information.
        
        Returns
        -------
        PortfolioBuilderSum
            Self for method chaining
        """
        self._components_info.clear()
        self._edges.clear()
        return self
        
    def from_paths(self, rows: List[Dict[str, Any]]) -> 'PortfolioBuilderSum':
        """
        Add multiple components from a list of path-based dictionaries.
        
        Parameters
        ----------
        rows : list of dict
            List of dictionaries containing path and weight information
            
        Returns
        -------
        PortfolioBuilderSum
            Self for method chaining
        """
        for row in rows:
            self.add_path(
                path=row['path'],
                portfolio_weight=row.get('portfolio_weight'),
                benchmark_weight=row.get('benchmark_weight'),
                component_type=row.get('component_type', 'auto'),
                is_overlay=row.get('is_overlay', False),
                name=row.get('name'),
                data=row.get('data')
            )
        return self
    
    def from_hierarchy(self, hierarchy: Dict[str, Any], base_path: str = "") -> 'PortfolioBuilderSum':
        """
        Add components from a hierarchical dictionary structure.
        
        Parameters
        ----------
        hierarchy : dict
            Nested dictionary structure defining the portfolio hierarchy
        base_path : str, default ""
            Base path prefix for this hierarchy level
            
        Returns
        -------
        PortfolioBuilderSum
            Self for method chaining
        """
        for key, value in hierarchy.items():
            current_path = f"{base_path}{self.delimiter}{key}" if base_path else key
            
            if isinstance(value, dict):
                if 'children' in value:
                    # This is a parent node with children
                    self.add_path(
                        path=current_path,
                        portfolio_weight=value.get('portfolio_weight'),
                        benchmark_weight=value.get('benchmark_weight'),
                        component_type='node',
                        is_overlay=value.get('is_overlay', False),
                        name=value.get('name'),
                        data=value.get('data')
                    )
                    # Recursively process children
                    self.from_hierarchy(value['children'], current_path)
                else:
                    # This is a leaf node
                    self.add_path(
                        path=current_path,
                        portfolio_weight=value.get('portfolio_weight'),
                        benchmark_weight=value.get('benchmark_weight'),
                        component_type='leaf',
                        is_overlay=value.get('is_overlay', False),
                        name=value.get('name'),
                        data=value.get('data')
                    )
            else:
                # Simple value - treat as portfolio weight for a leaf
                self.add_path(
                    path=current_path,
                    portfolio_weight=value,
                    component_type='leaf'
                )
        
        return self
        
    def build(self) -> 'PortfolioGraph':
        """
        Build the PortfolioGraph using sum-based weight aggregation.
        
        Returns
        -------
        PortfolioGraph
            Constructed portfolio graph with sum-aggregated weights
        """
        # Import here to avoid circular imports
        from .graph import PortfolioGraph
        from .components import PortfolioNode, PortfolioLeaf
        from .metrics import ScalarMetric
        
        # Create the graph
        graph = PortfolioGraph(root_id=self.root_id, metric_store=self.metric_store)
        
        # Add root component if not already added
        if self.root_id not in self._components_info:
            root_component = PortfolioNode(component_id=self.root_id, name='Portfolio Root')
            graph.add_component(root_component)
            
        # Add all components to the graph
        for component_id, comp_info in self._components_info.items():
            if component_id == self.root_id:
                continue  # Root already added
                
            # Determine component type
            has_children = any(parent_id == component_id for parent_id, _ in self._edges)
            
            if comp_info['component_type'] == 'auto':
                comp_type = 'node' if has_children else 'leaf'
            else:
                comp_type = comp_info['component_type']
                
            # Create appropriate component
            if comp_type == 'leaf':
                component = PortfolioLeaf(
                    component_id=component_id,
                    name=comp_info['name']
                )
            else:
                component = PortfolioNode(
                    component_id=component_id, 
                    name=comp_info['name']
                )
            
            # Store additional data in component metadata
            if comp_info['data']:
                component.metadata.update(comp_info['data'])
            
            graph.add_component(component)
            
            # Store provided weights (for leaves) or None (for nodes to be calculated)
            if comp_info['portfolio_weight'] is not None:
                graph.metric_store.set_metric(
                    component_id, 'portfolio_weight', 
                    ScalarMetric(comp_info['portfolio_weight'])
                )
            if comp_info['benchmark_weight'] is not None:
                graph.metric_store.set_metric(
                    component_id, 'benchmark_weight',
                    ScalarMetric(comp_info['benchmark_weight'])
                )
                
            # Store overlay flag
            if comp_info['is_overlay']:
                graph.metric_store.set_metric(
                    component_id, 'is_overlay',
                    ScalarMetric(1.0)
                )
        
        # Create edges and connect to root if needed
        self._create_graph_edges(graph)
        
        # Use WeightPathAggregatorSum to calculate parent weights
        self._calculate_aggregated_weights(graph)
        
        # Apply normalization if requested
        if self.normalize:
            self._normalize_to_unit_portfolio(graph)
        
        return graph
    
    def _create_graph_edges(self, graph: 'PortfolioGraph') -> None:
        """Create edges in the graph and ensure connectivity to root."""
        # Add all specified edges
        for parent_id, child_id in self._edges:
            if not graph.add_edge(parent_id, child_id):
                # Handle case where parent_id doesn't exist - connect to root
                if parent_id not in graph.components:
                    graph.add_edge(self.root_id, child_id)
                else:
                    raise ValueError(f"Failed to add edge from {parent_id} to {child_id}")
        
        # Ensure all top-level components are connected to root
        top_level_components = set()
        for parent_id, child_id in self._edges:
            # Find components that are not children of anything (except possibly root)
            if parent_id not in [edge[1] for edge in self._edges if edge[0] != self.root_id]:
                top_level_components.add(parent_id)
        
        for comp_id in top_level_components:
            if comp_id != self.root_id and comp_id in graph.components:
                # Connect to root if not already connected
                if not any(edge[0] == self.root_id and edge[1] == comp_id for edge in self._edges):
                    graph.add_edge(self.root_id, comp_id)
                    
    def _calculate_aggregated_weights(self, graph: 'PortfolioGraph') -> None:
        """Calculate parent weights using WeightPathAggregatorSum logic."""
        from .metrics import ScalarMetric
        
        # Simple bottom-up calculation: parent = sum of children
        # Process in reverse topological order to ensure children are processed first
        processed = set()
        
        def process_component(comp_id: str):
            if comp_id in processed:
                return
                
            # Get children from adjacency list
            children = list(graph._adjacency_list.get(comp_id, set()))
            
            # Process children first
            for child_id in children:
                process_component(child_id)
            
            # Calculate this component's weight as sum of children
            if children:  # Only for non-leaf nodes
                for weight_type in ['portfolio_weight', 'benchmark_weight']:
                    total_weight = 0.0
                    has_weight = False
                    
                    for child_id in children:
                        child_weight_metric = graph.metric_store.get_metric(child_id, weight_type)
                        if child_weight_metric is not None:
                            child_weight = child_weight_metric.value()
                            
                            # Include/exclude overlays based on configuration
                            is_overlay_metric = graph.metric_store.get_metric(child_id, 'is_overlay')
                            is_overlay = is_overlay_metric is not None and is_overlay_metric.value() > 0
                            
                            if not is_overlay or self.include_overlays_in_aggregation:
                                total_weight += child_weight
                                has_weight = True
                    
                    if has_weight:
                        graph.metric_store.set_metric(comp_id, weight_type, ScalarMetric(total_weight))
            
            processed.add(comp_id)
        
        # Start from root and process all components
        process_component(self.root_id)
    
    def _normalize_to_unit_portfolio(self, graph: 'PortfolioGraph') -> None:
        """
        Normalize the portfolio so that the root weight equals 1.0.
        
        This scales only non-overlay weights proportionally so that the sum of all 
        non-overlay leaf weights equals 1.0. Overlay weights remain unchanged.
        """
        from .metrics import ScalarMetric
        
        # Get current root weight (sum of all non-overlay leaves)
        root_weight_metric = graph.metric_store.get_metric(self.root_id, 'portfolio_weight')
        if root_weight_metric is None:
            return  # Nothing to normalize
        
        current_root_weight = root_weight_metric.value()
        if current_root_weight == 0.0:
            return  # Can't normalize zero weight
        
        # Calculate scaling factor to make root = 1.0
        scaling_factor = 1.0 / current_root_weight
        
        # Apply scaling only to non-overlay components
        for component_id in graph.components:
            # Check if this is an overlay
            is_overlay_metric = graph.metric_store.get_metric(component_id, 'is_overlay')
            is_overlay = is_overlay_metric is not None and is_overlay_metric.value() > 0
            
            if not is_overlay:  # Only scale non-overlay components
                for weight_type in ['portfolio_weight', 'benchmark_weight']:
                    weight_metric = graph.metric_store.get_metric(component_id, weight_type)
                    if weight_metric is not None:
                        current_weight = weight_metric.value()
                        normalized_weight = current_weight * scaling_factor
                        graph.metric_store.set_metric(
                            component_id, weight_type, 
                            ScalarMetric(normalized_weight)
                        )
        
        # Store normalization metadata
        graph.metric_store.set_metric(
            self.root_id, 'normalization_factor', 
            ScalarMetric(scaling_factor)
        )
        graph.metric_store.set_metric(
            self.root_id, 'original_root_weight',
            ScalarMetric(current_root_weight)
        )