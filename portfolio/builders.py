"""
Portfolio Hierarchy Builders
============================

Builder classes for constructing complex hierarchical portfolios with
enhanced features like path-based addressing, template systems, and
automatic weight normalization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .graph import PortfolioGraph
    from .metrics import MetricStore

from .metrics import InMemoryMetricStore, WeightPathAggregator, WeightPathAggregatorSum


class PortfolioBuilder:
    """
    Builder class for constructing hierarchical portfolios with automatic weight normalization.
    
    Core Features:
    - Path-based component addition with hierarchical structure
    - Automatic weight normalization for WeightPathAggregator compatibility  
    - Support for overlay strategies and short positions
    - Portfolio + benchmark weight tracking
    - Hierarchical dict and path-based data input methods
    """
    
    def __init__(self,
                 *,
                 allow_shorts: bool = False,
                 auto_normalize_hierarchy: bool = True,
                 overlay_weight_mode: Literal["dual", "allocation_only"] = "dual",
                 tol: float = 1e-8,
                 delimiter: str = "/",
                 root_id: str = 'portfolio',
                 metric_store: Optional['MetricStore'] = None):
        """
        Initialize the portfolio builder.
        
        Parameters
        ----------
        allow_shorts : bool, default False
            Whether to allow negative weights
        auto_normalize_hierarchy : bool, default True
            Enable automatic weight normalization to ensure descendants are
            properly normalized as their share of parents for WeightPathAggregator
        overlay_weight_mode : {"dual", "allocation_only"}, default "dual"
            How to handle overlay position weights:
            - "dual": Track both allocation (0.0) and operational weights
            - "allocation_only": Treat overlays as regular allocations
        tol : float, default 1e-8
            Numerical tolerance for validations and comparisons
        delimiter : str, default "/"
            Path delimiter character for hierarchical paths
        root_id : str, default 'portfolio'
            ID for the root component
        metric_store : MetricStore, optional
            Metric store to use. If None, creates new InMemoryMetricStore
        """
        # Configuration parameters
        self.allow_shorts = allow_shorts
        self.auto_normalize_hierarchy = auto_normalize_hierarchy
        self.overlay_weight_mode = overlay_weight_mode
        self.tol = tol
        self.delimiter = delimiter
        self.root_id = root_id
        self.metric_store = metric_store or InMemoryMetricStore()
        
        # Internal storage
        self._components_info = {}  # Store component info before building graph
        self._edges = []  # Store edge relationships
        
    def add_path(self, 
                 path: str, 
                 portfolio_weight: Optional[float] = None,
                 benchmark_weight: Optional[float] = 0.0,
                 component_type: Optional[str] = None,
                 name: Optional[str] = None,
                 data: Optional[Dict[str, Any]] = None,
                 is_overlay: bool = False) -> 'PortfolioBuilder':
        """
        Add a component at the specified path, creating intermediate nodes as needed.
        
        Parameters
        ----------
        path : str
            Full path to the component (e.g., "equity/us/large_cap/tech/AAPL")
        portfolio_weight : float, optional
            Portfolio weight for this component
        benchmark_weight : float, default 0.0
            Benchmark weight for this component
        component_type : str, optional
            Type of component ('node' or 'leaf'). If None, inferred from path structure
        name : str, optional
            Display name for component. If None, uses last part of path
        data : dict, optional
            Additional data to store for this component
        is_overlay : bool, default False
            Whether this component is an overlay strategy (uses parent operational weight)
            
        Returns
        -------
        PortfolioBuilder
            Self for method chaining
        """
        path_parts = [part.strip() for part in path.split('/') if part.strip()]
        
        if not path_parts:
            raise ValueError("Invalid path: empty or whitespace-only")
        
        # Create all components in the path
        for i, part in enumerate(path_parts):
            current_path = '/'.join(path_parts[:i+1])
            is_leaf = (i == len(path_parts) - 1)
            
            # Skip if component already exists
            if current_path in self._components_info:
                continue
            
            # Determine component type
            if component_type is not None and is_leaf:
                comp_type = component_type
            else:
                comp_type = 'leaf' if is_leaf else 'node'
            
            # Determine component name
            if name is not None and is_leaf:
                comp_name = name
            else:
                comp_name = part
            
            # Store component info
            self._components_info[current_path] = {
                'id': current_path,
                'name': comp_name,
                'type': comp_type,
                'portfolio_weight': portfolio_weight,
                'benchmark_weight': benchmark_weight,
                'data': data if is_leaf else None,
                'path_parts': path_parts[:i+1],
                'is_overlay': is_overlay if is_leaf else False
            }
            
            # Add edge relationship (except for root)
            if i > 0:
                parent_path = '/'.join(path_parts[:i])
                self._edges.append((parent_path, current_path))
        
        return self
    
    def from_hierarchy(self, data: Dict[str, Any], *, weight_mode: Optional[str] = None) -> 'PortfolioBuilder':
        """
        Build PortfolioGraph from hierarchical dictionary structure.
        
        Parameters
        ----------
        data : dict
            Hierarchical dictionary with portfolio + benchmark structure.
            Each node should contain:
            - component_id : str
            - type : "node" | "leaf"
            - portfolio_weight : float
            - portfolio_weight_type : "relative" | "absolute"
            - benchmark_weight : float  
            - benchmark_weight_type : "relative" | "absolute"
            - children : list, optional (for nodes)
            - meta : dict, optional
            - is_overlay : bool, optional (default False)
                Whether this component is an overlay strategy
        weight_mode : str, optional
            Override weight mode for processing
            
        Returns
        -------
        PortfolioBuilder
            Self for method chaining, call build() to create PortfolioGraph
        """
        self._process_hierarchy_node(data, path_prefix="")
        return self
    
    def from_paths(self, rows: List[Dict[str, Any]], *, weight_mode: Optional[str] = None) -> 'PortfolioBuilder':
        """
        Build PortfolioGraph from path-based row data.
        
        Parameters
        ----------
        rows : list of dict
            List of row dictionaries, each containing:
            - type : "node" | "leaf"
            - path : str (e.g., "total/equity/developed/us")
            - component_id : str, optional (defaults to last path segment)
            - portfolio_weight : float, optional
            - portfolio_weight_type : "relative" | "absolute", optional
            - benchmark_weight : float, optional
            - benchmark_weight_type : "relative" | "absolute", optional
            - meta : dict, optional
            - is_overlay : bool, optional (default False)
                Whether this component is an overlay strategy
        weight_mode : str, optional
            Override weight mode for processing
            
        Returns
        -------
        PortfolioBuilder
            Self for method chaining, call build() to create PortfolioGraph
        """
        self._process_path_rows(rows)
        return self
    
    def clear(self) -> 'PortfolioBuilder':
        """
        Clear all accumulated component and edge information for a fresh start.
        
        Returns
        -------
        PortfolioBuilder
            Self for method chaining
        """
        self._components_info.clear()
        self._edges.clear()
        return self
    
    
    def build(self) -> 'PortfolioGraph':
        """
        Build the PortfolioGraph from the accumulated component and edge information.
        
        Returns
        -------
        PortfolioGraph
            Constructed portfolio graph
        """
        # Import here to avoid circular imports
        from .graph import PortfolioGraph
        
        # Validate basic constraints
        self._validate_shorts()
        
        # Auto-normalize weights (if enabled)
        self._auto_normalize_weights()
        
        # Create the graph
        graph = PortfolioGraph(root_id=self.root_id, metric_store=self.metric_store)
        
        # Add root component if not already added
        if self.root_id not in self._components_info:
            from .components import PortfolioNode
            from .metrics import ScalarMetric
            root_component = PortfolioNode(component_id=self.root_id, name='Portfolio Root')
            graph.add_component(root_component)
            graph.metric_store.set_metric(self.root_id, 'portfolio_weight', ScalarMetric(1.0))
            graph.metric_store.set_metric(self.root_id, 'benchmark_weight', ScalarMetric(1.0))
        
        # Ensure all top-level components are connected to root
        top_level_components = set()
        for parent_id, child_id in self._edges:
            if parent_id not in [edge[1] for edge in self._edges]:  # parent_id is not a child of anything
                top_level_components.add(parent_id)
        
        for comp_id in top_level_components:
            if comp_id != self.root_id:
                # Add edge from root to this top-level component if not already connected
                root_connected = any(edge[0] == self.root_id and edge[1] == comp_id for edge in self._edges)
                if not root_connected:
                    self._edges.append((self.root_id, comp_id))
        
        # Create components
        for component_id, info in self._components_info.items():
            from .components import PortfolioNode, PortfolioLeaf
            from .metrics import ScalarMetric, SeriesMetric
            
            if info['type'] == 'leaf':
                component = PortfolioLeaf(component_id=component_id, name=info['name'])
            else:
                component = PortfolioNode(component_id=component_id, name=info['name'])
            
            graph.add_component(component)
            
            # Set overlay property if applicable
            if info.get('is_overlay', False):
                component.is_overlay = True
            
            # Set weights (both absolute and relative if calculated)
            if info['portfolio_weight'] is not None:
                graph.metric_store.set_metric(component_id, 'portfolio_weight', 
                                            ScalarMetric(info['portfolio_weight']))
            if info['benchmark_weight'] is not None:
                graph.metric_store.set_metric(component_id, 'benchmark_weight', 
                                            ScalarMetric(info['benchmark_weight']))
            
            # Set relative weights if calculated during auto-normalization
            if 'portfolio_relative_weight' in info:
                graph.metric_store.set_metric(component_id, 'portfolio_relative_weight',
                                            ScalarMetric(info['portfolio_relative_weight']))
            if 'benchmark_relative_weight' in info:
                graph.metric_store.set_metric(component_id, 'benchmark_relative_weight',
                                            ScalarMetric(info['benchmark_relative_weight']))
            
            # Set operational weights for overlays if using dual mode
            if info.get('is_overlay', False) and self.overlay_weight_mode == "dual":
                if 'portfolio_operational_weight' in info:
                    graph.metric_store.set_metric(component_id, 'portfolio_operational_weight',
                                                ScalarMetric(info['portfolio_operational_weight']))
                if 'benchmark_operational_weight' in info:
                    graph.metric_store.set_metric(component_id, 'benchmark_operational_weight',
                                                ScalarMetric(info['benchmark_operational_weight']))
                if 'portfolio_operational_relative' in info:
                    graph.metric_store.set_metric(component_id, 'portfolio_operational_relative',
                                                ScalarMetric(info['portfolio_operational_relative']))
                if 'benchmark_operational_relative' in info:
                    graph.metric_store.set_metric(component_id, 'benchmark_operational_relative',
                                                ScalarMetric(info['benchmark_operational_relative']))
            
            # Set original weights if auto-normalized
            if info.get('auto_normalized', False):
                if 'portfolio_weight_original' in info:
                    graph.metric_store.set_metric(component_id, 'portfolio_weight_original',
                                                ScalarMetric(info['portfolio_weight_original']))
                if 'benchmark_weight_original' in info:
                    graph.metric_store.set_metric(component_id, 'benchmark_weight_original',
                                                ScalarMetric(info['benchmark_weight_original']))
                
                # Store weight type metadata
                from .metrics import Metric
                class StringMetric(Metric):
                    def __init__(self, value):
                        self._value = value
                    def value(self, when=None):
                        return self._value
                    def copy(self):
                        return StringMetric(self._value)
                        
                if 'portfolio_weight_type_original' in info:
                    graph.metric_store.set_metric(component_id, 'portfolio_weight_type_original',
                                                StringMetric(info['portfolio_weight_type_original']))
                if 'benchmark_weight_type_original' in info:
                    graph.metric_store.set_metric(component_id, 'benchmark_weight_type_original', 
                                                StringMetric(info['benchmark_weight_type_original']))
                    
                # Add normalization flag
                graph.metric_store.set_metric(component_id, 'auto_normalized',
                                            ScalarMetric(1.0))  # Use 1.0 as True
            
            # Set additional data
            if info['data']:
                for key, value in info['data'].items():
                    if isinstance(value, pd.Series):
                        graph.metric_store.set_metric(component_id, key, SeriesMetric(value))
                    elif isinstance(value, (int, float)):
                        graph.metric_store.set_metric(component_id, key, ScalarMetric(value))
                    else:
                        # Store strings and other types as a simple metric
                        from .metrics import Metric
                        class StringMetric(Metric):
                            def __init__(self, value):
                                self._value = value
                            def value(self, when=None):
                                return self._value
                            def copy(self):
                                return StringMetric(self._value)
                        graph.metric_store.set_metric(component_id, key, StringMetric(value))
        
        # Add edges
        for parent_id, child_id in self._edges:
            graph.add_edge(parent_id, child_id)
        
        # Validate the graph
        is_valid, issues = graph.validate_structure()
        if not is_valid:
            print(f"Warning: Graph validation issues found: {issues}")
        
        # Add utility methods to the graph for weight retrieval
        self._add_weight_utility_methods(graph)
        
        return graph
    
    def _add_weight_utility_methods(self, graph: 'PortfolioGraph') -> None:
        """Add utility methods to PortfolioGraph for weight retrieval."""
        
        def get_original_weight(component_id: str, weight_type: str = 'portfolio') -> Optional[float]:
            """Get original (pre-normalization) weight for a component."""
            metric_name = f'{weight_type}_weight_original'
            metric = graph.metric_store.get_metric(component_id, metric_name)
            return metric.value() if metric else None
        
        def get_normalized_weight(component_id: str, weight_type: str = 'portfolio') -> Optional[float]:
            """Get normalized weight for a component."""
            metric_name = f'{weight_type}_weight'
            metric = graph.metric_store.get_metric(component_id, metric_name)
            return metric.value() if metric else None
        
        def was_auto_normalized(component_id: str) -> bool:
            """Check if component weights were auto-normalized."""
            metric = graph.metric_store.get_metric(component_id, 'auto_normalized')
            return metric.value() == 1.0 if metric else False
        
        def get_weight_summary(component_id: str, weight_type: str = 'portfolio') -> Dict[str, Any]:
            """Get comprehensive weight summary for a component."""
            return {
                'component_id': component_id,
                'weight_type': weight_type,
                'original_weight': get_original_weight(component_id, weight_type),
                'normalized_weight': get_normalized_weight(component_id, weight_type),
                'relative_weight': graph.metric_store.get_metric(component_id, f'{weight_type}_relative_weight').value()
                    if graph.metric_store.get_metric(component_id, f'{weight_type}_relative_weight') else None,
                'auto_normalized': was_auto_normalized(component_id),
                'is_overlay': hasattr(graph.components.get(component_id, None), 'is_overlay') 
                    and getattr(graph.components[component_id], 'is_overlay', False)
            }
        
        # Attach methods to graph instance
        graph.get_original_weight = get_original_weight
        graph.get_normalized_weight = get_normalized_weight  
        graph.was_auto_normalized = was_auto_normalized
        graph.get_weight_summary = get_weight_summary
    
    
    
    def _clear_internal_state(self) -> None:
        """Clear all internal state for a fresh build."""
        self._components_info.clear()
        self._edges.clear()
    
    def _process_hierarchy_node(self, node_data: Dict[str, Any], path_prefix: str) -> None:
        """Process a hierarchical node and its children recursively."""
        component_id = node_data['component_id']
        full_path = f"{path_prefix}/{component_id}" if path_prefix else component_id
        
        # Store component info
        self._components_info[full_path] = {
            'id': component_id,
            'name': node_data.get('name', component_id),
            'type': node_data.get('type', 'node'),
            'portfolio_weight': node_data.get('portfolio_weight'),
            'portfolio_weight_type': node_data.get('portfolio_weight_type', 'absolute'),
            'benchmark_weight': node_data.get('benchmark_weight'),
            'benchmark_weight_type': node_data.get('benchmark_weight_type', 'absolute'),
            'data': node_data.get('meta'),
            'path_parts': full_path.split(self.delimiter),
            'is_overlay': node_data.get('is_overlay', False)
        }
        
        # Process children if they exist
        if 'children' in node_data:
            for child_data in node_data['children']:
                child_component_id = child_data['component_id']
                child_path = f"{full_path}/{child_component_id}"
                
                # Add edge relationship
                self._edges.append((full_path, child_path))
                
                # Recursively process child
                self._process_hierarchy_node(child_data, full_path)
    
    def _process_path_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Process path-based row data."""
        for row in rows:
            path = row['path']
            component_id = row.get('component_id', path.split(self.delimiter)[-1])
            
            # Store component info
            self._components_info[path] = {
                'id': component_id,
                'name': component_id,
                'type': row.get('type', 'leaf'),
                'portfolio_weight': row.get('portfolio_weight'),
                'portfolio_weight_type': row.get('portfolio_weight_type', 'absolute'),
                'benchmark_weight': row.get('benchmark_weight'),
                'benchmark_weight_type': row.get('benchmark_weight_type', 'absolute'),
                'data': row.get('meta'),
                'path_parts': path.split(self.delimiter),
                'is_overlay': row.get('is_overlay', False)
            }
            
            # Create edges for intermediate nodes
            path_parts = path.split(self.delimiter)
            for i in range(1, len(path_parts)):
                parent_path = self.delimiter.join(path_parts[:i])
                child_path = self.delimiter.join(path_parts[:i+1])
                edge = (parent_path, child_path)
                if edge not in self._edges:
                    self._edges.append(edge)
    
    def _validate_shorts(self) -> None:
        """Validate that shorts are allowed if present."""
        if not self.allow_shorts:
            for comp_path, comp_info in self._components_info.items():
                for weight_side in ['portfolio', 'benchmark']:
                    weight = comp_info.get(f'{weight_side}_weight')
                    if weight is not None and weight < -self.tol:
                        raise ValueError(
                            f"Negative weight not allowed: {comp_path} {weight_side}={weight}"
                        )
    
    def _auto_normalize_weights(self) -> None:
        """
        Auto-normalize weights to ensure WeightPathAggregator consistency.
        
        Phase 1: Bottom-up aggregation - calculate parent weights from children
        Phase 2: Top-down relative weight calculation for multiplication consistency
        """
        if not self.auto_normalize_hierarchy:
            return
        
        # Store original weights before normalization
        self._preserve_original_weights()
            
        # Phase 1: Bottom-up weight aggregation (leaves to root)
        topo_order = self._get_topological_order()
        
        for component_id in reversed(topo_order):  # Start from leaves
            if self._is_leaf_component(component_id):
                continue  # Leaves keep their target weights
            
            # Calculate parent weight from core children (excluding overlays)
            self._calculate_parent_core_weights(component_id)
        
        # Phase 2: Top-down relative weight calculation (root to leaves)
        for component_id in topo_order:  # Start from root
            self._calculate_relative_weights(component_id)
    
    
    def _get_topological_order(self) -> List[str]:
        """Get topological ordering of components for weight propagation."""
        # Build adjacency map
        adj_map = {}
        in_degree = {}
        
        # Initialize all components
        for comp_id in self._components_info.keys():
            adj_map[comp_id] = []
            in_degree[comp_id] = 0
        
        # Build edges and calculate in-degrees
        for parent_id, child_id in self._edges:
            if parent_id in adj_map:
                adj_map[parent_id].append(child_id)
            if child_id in in_degree:
                in_degree[child_id] += 1
        
        # Kahn's algorithm for topological sort
        queue = [comp_id for comp_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in adj_map.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _is_leaf_component(self, component_id: str) -> bool:
        """Check if component is a leaf (has no children)."""
        for parent_id, child_id in self._edges:
            if parent_id == component_id:
                return False
        return True
    
    def _is_root_component(self, component_id: str) -> bool:
        """Check if component is a root (has no parents)."""
        for parent_id, child_id in self._edges:
            if child_id == component_id:
                return False
        return True
    
    def _get_direct_children(self, component_id: str) -> List[str]:
        """Get direct children of a component."""
        children = []
        for parent_id, child_id in self._edges:
            if parent_id == component_id:
                children.append(child_id)
        return children
    
    def _calculate_parent_core_weights(self, parent_id: str) -> None:
        """Calculate parent weights from core (non-overlay) children."""
        children = self._get_direct_children(parent_id)
        
        for weight_side in ['portfolio', 'benchmark']:
            weight_key = f'{weight_side}_weight'
            core_sum = 0.0
            
            for child_id in children:
                if child_id not in self._components_info:
                    continue
                
                child_info = self._components_info[child_id]
                child_weight = child_info.get(weight_key, 0.0)
                
                # Include in core sum (exclude overlays, include all weights including shorts)
                if child_info.get('is_overlay', False):
                    # Overlays don't contribute to parent allocation weight
                    continue
                elif child_weight is not None:
                    # Always include all weights (including shorts) in parent calculation
                    core_sum += child_weight
            
            # Set parent weight to core children sum
            if parent_id in self._components_info:
                self._components_info[parent_id][weight_key] = core_sum
    
    def _calculate_relative_weights(self, parent_id: str) -> None:
        """Calculate relative weights for children and replace absolute weights with relative weights."""
        children = self._get_direct_children(parent_id)
        if not children:
            return
        
        for weight_side in ['portfolio', 'benchmark']:
            weight_key = f'{weight_side}_weight'
            relative_key = f'{weight_side}_relative_weight'
            operational_key = f'{weight_side}_operational_weight'
            
            parent_info = self._components_info.get(parent_id, {})
            
            # Get the current absolute weight of parent (this was calculated in bottom-up phase)
            parent_weight = parent_info.get(weight_key, 0.0)
            if parent_weight is None or parent_weight == 0.0:
                parent_weight = 1.0 if self._is_root_component(parent_id) else 0.0
            
            # Calculate the sum of children's absolute weights (core only, excluding overlays)
            core_children_sum = 0.0
            for child_id in children:
                if child_id not in self._components_info:
                    continue
                child_info = self._components_info[child_id]
                if not child_info.get('is_overlay', False):
                    child_weight = child_info.get(weight_key, 0.0)
                    if child_weight is not None:
                        core_children_sum += child_weight
            
            # Convert children to relative weights (as share of their sum, not parent weight)
            for child_id in children:
                if child_id not in self._components_info:
                    continue
                
                child_info = self._components_info[child_id]
                child_absolute = child_info.get(weight_key, 0.0)
                
                if child_info.get('is_overlay', False):
                    # Overlay position handling
                    if self.overlay_weight_mode == "dual":
                        # Dual mode: allocation weight = 0, operational weight = parent weight
                        relative_weight = 0.0
                        child_info[relative_key] = relative_weight
                        child_info[weight_key] = relative_weight
                        
                        # Find first non-overlay ancestor for operational weight
                        operational_weight = self._find_operational_parent_weight(parent_id, weight_side)
                        child_info[operational_key] = operational_weight
                        child_info[f'{weight_side}_operational_relative'] = 1.0
                    else:
                        # Allocation only mode: treat as regular position
                        if core_children_sum != 0:
                            relative_weight = child_absolute / core_children_sum
                        else:
                            relative_weight = 0.0
                        child_info[relative_key] = relative_weight
                        child_info[weight_key] = relative_weight
                else:
                    # Regular position: calculate relative weight as share of core children sum
                    if core_children_sum != 0:
                        relative_weight = child_absolute / core_children_sum
                    else:
                        relative_weight = 0.0
                    
                    child_info[relative_key] = relative_weight
                    child_info[weight_key] = relative_weight  # Replace absolute with relative
    
    def _find_operational_parent_weight(self, component_id: str, weight_side: str) -> float:
        """Find the first non-overlay ancestor's weight for operational weight calculation."""
        weight_key = f'{weight_side}_weight'
        current_id = component_id
        
        while current_id:
            current_info = self._components_info.get(current_id, {})
            
            # If this component is not an overlay and has a weight, use it
            if not current_info.get('is_overlay', False):
                weight = current_info.get(weight_key, 0.0)
                if weight is not None and weight != 0.0:
                    return weight
            
            # Move to parent
            parent_edges = [(p, c) for p, c in self._edges if c == current_id]
            if parent_edges:
                current_id = parent_edges[0][0]  # Take first parent
            else:
                break
        
        # If no non-overlay parent found, return 0
        return 0.0
    
    def _preserve_original_weights(self) -> None:
        """Preserve original input weights before auto-normalization."""
        for component_id, info in self._components_info.items():
            # Store original weights with audit trail
            if 'portfolio_weight' in info and info['portfolio_weight'] is not None:
                info['portfolio_weight_original'] = info['portfolio_weight']
                info['portfolio_weight_type_original'] = info.get('portfolio_weight_type', 'absolute')
            
            if 'benchmark_weight' in info and info['benchmark_weight'] is not None:
                info['benchmark_weight_original'] = info['benchmark_weight']  
                info['benchmark_weight_type_original'] = info.get('benchmark_weight_type', 'absolute')
                
            # Add normalization metadata
            info['auto_normalized'] = True
    
    


def create_graph_from_dataframe(df: pd.DataFrame, root_id: str = 'portfolio', metric_store: Optional['MetricStore'] = None) -> 'PortfolioGraph':
    """
    Create PortfolioGraph from DataFrame with parent-child relationships
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing portfolio hierarchy with columns:
        - parent: Parent component IDs
        - child: Child component IDs
        - weight: Component weights
        - scaling_factor: Scaling factors (optional)
        - forward_returns: Forward returns (optional)
        - forward_risk: Forward risk (optional)
        - forward_specific_return: Forward specific returns (optional)
        - forward_specific_risk: Forward specific risk (optional)
    root_id : str, default 'portfolio'
        ID for the root component
    metric_store : MetricStore, optional
        Metric store to use. If None, creates new InMemoryMetricStore
    
    Returns
    -------
    PortfolioGraph
        Constructed portfolio graph
    """
    # Import here to avoid circular imports
    from .graph import PortfolioGraph
    
    # Validate required columns
    required_columns = ['parent', 'child', 'weight']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Create new graph
    graph = PortfolioGraph(root_id=root_id, metric_store=metric_store)
    
    # Get all unique component IDs
    all_component_ids = set(df['parent'].unique()) | set(df['child'].unique())
    
    # Create components
    for component_id in all_component_ids:
        # Determine if this is a leaf (appears only in child column)
        is_leaf = component_id not in df['parent'].values
        
        # Get metrics for this component from the first row where it appears
        component_rows = df[(df['parent'] == component_id) | (df['child'] == component_id)]
        if not component_rows.empty:
            row = component_rows.iloc[0]
            
            # Extract optional metrics
            scaling_factor = row.get('scaling_factor', 1.0)
            forward_returns = row.get('forward_returns', None)
            forward_risk = row.get('forward_risk', None)
            forward_specific_returns = row.get('forward_specific_return', None)
            forward_specific_risk = row.get('forward_specific_risk', None)
            weight = row.get('weight', 0.0)
            
            # Create component
            from .components import PortfolioLeaf, PortfolioNode
            from .metrics import ScalarMetric
            
            if is_leaf:
                component = PortfolioLeaf(
                    component_id=component_id,
                    name=component_id,
                    component_type="security"
                )
            else:
                component = PortfolioNode(
                    component_id=component_id,
                    name=component_id,
                    component_type="portfolio"
                )
            
            graph.add_component(component)
            
            # Store metrics in metric store
            if forward_returns is not None:
                graph.metric_store.set_metric(component_id, 'forward_returns', ScalarMetric(forward_returns))
            if forward_risk is not None:
                graph.metric_store.set_metric(component_id, 'forward_risk', ScalarMetric(forward_risk))
            if weight is not None:
                graph.metric_store.set_metric(component_id, 'weight', ScalarMetric(weight))
            if scaling_factor is not None:
                graph.metric_store.set_metric(component_id, 'scaling_factor', ScalarMetric(scaling_factor))
            if forward_specific_returns is not None:
                graph.metric_store.set_metric(component_id, 'forward_specific_returns', ScalarMetric(forward_specific_returns))
            if forward_specific_risk is not None:
                graph.metric_store.set_metric(component_id, 'forward_specific_risk', ScalarMetric(forward_specific_risk))
    
    # Create edges
    for _, row in df.iterrows():
        parent_id = row['parent']
        child_id = row['child']
        weight = row.get('weight', 0.0)
        
        # Store allocation tilt in metric store if non-zero
        if weight != 0.0:
            from .metrics import ScalarMetric
            edge_key = f"{parent_id}->{child_id}"
            graph.metric_store.set_metric(edge_key, 'allocation_tilt', ScalarMetric(weight))
        
        if not graph.add_edge(parent_id, child_id):
            raise ValueError(f"Failed to add edge from {parent_id} to {child_id}")
    
    # Set root (component with no parents)
    potential_roots = [cid for cid, component in graph.components.items() 
                      if len(component.parent_ids) == 0]
    if potential_roots:
        graph.root_id = potential_roots[0]
    
    return graph


def create_portfolio_graph_class_methods():
    """Create enhanced construction methods for PortfolioGraph class."""
    
    @classmethod
    def from_hierarchy_df(cls, 
                         df: pd.DataFrame, 
                         path_column: str = 'path',
                         portfolio_weight_column: str = 'portfolio_weight',
                         benchmark_weight_column: str = 'benchmark_weight',
                         name_column: Optional[str] = None,
                         data_columns: Optional[List[str]] = None,
                         overlay_column: Optional[str] = None,
                         root_id: str = 'portfolio') -> 'PortfolioGraph':
        """
        Create PortfolioGraph from DataFrame with hierarchical paths.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing hierarchy data with columns:
            - path: Full hierarchical path (e.g., "equity/us/large_cap/tech/AAPL")
            - portfolio_weight: Portfolio weights
            - benchmark_weight: Benchmark weights
            - Additional data columns as specified
        path_column : str, default 'path'
            Name of column containing hierarchical paths
        portfolio_weight_column : str, default 'portfolio_weight'
            Name of column containing portfolio weights
        benchmark_weight_column : str, default 'benchmark_weight'
            Name of column containing benchmark weights
        name_column : str, optional
            Name of column containing display names
        data_columns : list of str, optional
            Additional columns to store as component data
        overlay_column : str, optional
            Name of column containing overlay flags (boolean values)
        root_id : str, default 'portfolio'
            ID for root component
            
        Returns
        -------
        PortfolioGraph
            Constructed portfolio graph
        """
        builder = PortfolioBuilder(root_id=root_id)
        
        for _, row in df.iterrows():
            path = row[path_column]
            portfolio_weight = row.get(portfolio_weight_column, None)
            benchmark_weight = row.get(benchmark_weight_column, 0.0)
            name = row.get(name_column, None) if name_column else None
            is_overlay = bool(row.get(overlay_column, False)) if overlay_column else False
            
            # Collect additional data
            data = {}
            if data_columns:
                for col in data_columns:
                    if col in row and pd.notna(row[col]):
                        data[col] = row[col]
            
            builder.add_path(path, 
                           portfolio_weight=portfolio_weight,
                           benchmark_weight=benchmark_weight,
                           name=name,
                           data=data if data else None,
                           is_overlay=is_overlay)
        
        return builder.build()
    
    @classmethod
    def from_dict(cls, 
                  config: Dict[str, Any],
                  root_id: Optional[str] = None) -> 'PortfolioGraph':
        """
        Create PortfolioGraph from configuration dictionary.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with structure:
            {
                'root_id': 'portfolio',
                'components': {
                    'component_id': {
                        'type': 'node'|'leaf',
                        'name': 'Display Name',
                        'portfolio_weight': float,
                        'benchmark_weight': float,
                        'children': ['child1', 'child2'],  # for nodes
                        'data': {...}  # additional data
                    }
                },
                'weight_hierarchy': {...},  # optional relative weights
                'templates': {...}  # optional templates
            }
        root_id : str, optional
            Override root ID from config
            
        Returns
        -------
        PortfolioGraph
            Constructed portfolio graph
        """
        builder_root_id = root_id or config.get('root_id', 'portfolio')
        builder = PortfolioBuilder(root_id=builder_root_id)
        
        # Add templates if provided
        if 'templates' in config:
            for template_name, template_data in config['templates'].items():
                builder.add_template(template_name, template_data)
        
        # Add weight hierarchy if provided
        if 'weight_hierarchy' in config:
            builder.add_relative_weights(config['weight_hierarchy'])
        
        # Add components if provided
        if 'components' in config:
            for comp_id, comp_data in config['components'].items():
                comp_type = comp_data.get('type', 'leaf')
                name = comp_data.get('name', comp_id)
                portfolio_weight = comp_data.get('portfolio_weight')
                benchmark_weight = comp_data.get('benchmark_weight', 0.0)
                data = comp_data.get('data')
                
                builder.add_path(comp_id,
                               portfolio_weight=portfolio_weight,
                               benchmark_weight=benchmark_weight,
                               component_type=comp_type,
                               name=name,
                               data=data)
                
                # Handle children relationships
                if 'children' in comp_data:
                    for child_id in comp_data['children']:
                        if child_id in config['components']:
                            child_data = config['components'][child_id]
                            child_type = child_data.get('type', 'leaf')
                            child_name = child_data.get('name', child_id)
                            child_portfolio_weight = child_data.get('portfolio_weight')
                            child_benchmark_weight = child_data.get('benchmark_weight', 0.0)
                            child_data_dict = child_data.get('data')
                            
                            builder.add_path(f"{comp_id}/{child_id}",
                                           portfolio_weight=child_portfolio_weight,
                                           benchmark_weight=child_benchmark_weight,
                                           component_type=child_type,
                                           name=child_name,
                                           data=child_data_dict)
        
        return builder.build()
    
    def select_components(self, pattern: str) -> List[str]:
        """
        Select components matching a hierarchical pattern.
        
        Parameters
        ----------
        pattern : str
            Pattern to match (supports * wildcards)
            Examples: "equity/*", "*/us/*", "equity/us/large_cap/*"
            
        Returns
        -------
        list of str
            Component IDs matching the pattern
        """
        import fnmatch
        
        # Get all component paths by reconstructing from parent relationships
        component_paths = {}
        
        def build_path(component_id, visited=None):
            if visited is None:
                visited = set()
            if component_id in visited:
                return component_id  # Circular reference, return just the ID
            visited.add(component_id)
            
            component = self.components.get(component_id)
            if not component:
                return component_id
            
            # Find parent path
            if component.parent_ids:
                parent_id = next(iter(component.parent_ids))  # Take first parent
                parent_path = build_path(parent_id, visited.copy())
                return f"{parent_path}/{component_id}" if parent_path != component_id else component_id
            else:
                return component_id
        
        # Build paths for all components
        for component_id in self.components:
            component_paths[component_id] = build_path(component_id)
        
        # Match against pattern
        matching_ids = []
        for component_id, path in component_paths.items():
            if fnmatch.fnmatch(path, pattern):
                matching_ids.append(component_id)
        
        return matching_ids
    
    def update_subtree_weights(self, 
                             root_component_id: str,
                             portfolio_scale: float = 1.0,
                             benchmark_scale: float = 1.0,
                             recursive: bool = True):
        """
        Update weights for a subtree of components.
        
        Parameters
        ----------
        root_component_id : str
            Root component of subtree to update
        portfolio_scale : float, default 1.0
            Scaling factor for portfolio weights
        benchmark_scale : float, default 1.0
            Scaling factor for benchmark weights
        recursive : bool, default True
            Whether to apply scaling recursively to all descendants
        """
        from .metrics import ScalarMetric
        
        if root_component_id not in self.components:
            raise ValueError(f"Component {root_component_id} not found")
        
        def update_component_weights(component_id):
            # Update portfolio weight
            port_metric = self.metric_store.get_metric(component_id, 'portfolio_weight')
            if port_metric:
                current_value = port_metric.value()
                new_value = current_value * portfolio_scale
                self.metric_store.set_metric(component_id, 'portfolio_weight', ScalarMetric(new_value))
            
            # Update benchmark weight
            bench_metric = self.metric_store.get_metric(component_id, 'benchmark_weight')
            if bench_metric:
                current_value = bench_metric.value()
                new_value = current_value * benchmark_scale
                self.metric_store.set_metric(component_id, 'benchmark_weight', ScalarMetric(new_value))
        
        # Update root component
        update_component_weights(root_component_id)
        
        # Update descendants if recursive
        if recursive:
            component = self.components[root_component_id]
            if not component.is_leaf():
                for child_id in component.get_all_children():
                    if child_id in self.components:
                        self.update_subtree_weights(child_id, portfolio_scale, benchmark_scale, recursive)
    
    return {
        'from_hierarchy_df': from_hierarchy_df,
        'from_dict': from_dict,
        'select_components': select_components,
        'update_subtree_weights': update_subtree_weights
    }