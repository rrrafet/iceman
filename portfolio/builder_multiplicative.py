"""
PortfolioBuilderMultiplicative - Simplified Multiplicative Portfolio Builder
=========================================================================

A dramatically simplified portfolio builder for multiplicative weight aggregation
where children represent relative weights (shares) of their immediate parent,
and risk decomposition works through path multiplication.
"""

from typing import Dict, List, Optional, Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import PortfolioGraph
    from .metrics import MetricStore

from .metrics import InMemoryMetricStore


class PortfolioBuilderMultiplicative:
    """
    Simplified multiplicative portfolio builder for relative weight hierarchies.
    
    Key Features:
    - Children store weights as relative shares of immediate parent
    - Root weight always equals 1.0 after normalization
    - Single-phase normalization (no complex 4-phase system)
    - Works with WeightPathAggregator for risk decomposition
    - Much simpler logic (~250 lines vs 1,000+)
    
    Use Cases:
    - Risk decomposition and factor attribution
    - Hierarchical allocation frameworks
    - When you need multiplicative path aggregation
    """
    
    def __init__(self,
                 *,
                 allow_shorts: bool = False,
                 normalize_to_relative: bool = True,
                 proportional_renormalize: bool = False,
                 weight_reconcile_policy: Literal['respect_children', 'respect_parent'] = 'respect_children',
                 delimiter: str = "/",
                 root_id: str = 'portfolio',
                 metric_store: Optional['MetricStore'] = None):
        """
        Initialize the simplified multiplicative portfolio builder.
        
        Parameters
        ----------
        allow_shorts : bool, default False
            Whether to allow negative weights
        normalize_to_relative : bool, default True
            Whether to convert absolute weights to relative weights.
            When True, children become relative shares of their immediate parent.
            When False, weights are stored as provided (absolute).
        proportional_renormalize : bool, default False
            Whether to proportionally renormalize weights before processing.
            When True, leaf weights are normalized (weights / sum(weights)) and
            node weights are calculated by aggregating normalized child weights.
            Overlay components preserve their original weights.
        weight_reconcile_policy : {'respect_children', 'respect_parent'}, default 'respect_children'
            Policy for handling conflicts between node weights and children weights:
            - 'respect_children': Node weight = sum of children (children are authoritative)
            - 'respect_parent': Children scaled proportionally to fit node weight (node is authoritative)
        delimiter : str, default "/"
            Path delimiter character for hierarchical paths
        root_id : str, default 'portfolio'
            ID for the root component
        metric_store : MetricStore, optional
            Metric store to use. If None, creates new InMemoryMetricStore
        """
        # Configuration
        self.allow_shorts = allow_shorts
        self.normalize_to_relative = normalize_to_relative
        self.proportional_renormalize = proportional_renormalize
        self.weight_reconcile_policy = weight_reconcile_policy
        self.delimiter = delimiter
        self.root_id = root_id
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
                 data: Optional[Dict[str, Any]] = None) -> 'PortfolioBuilderMultiplicative':
        """
        Add a component to the portfolio hierarchy using path notation.
        
        Parameters
        ----------
        path : str
            Hierarchical path (e.g., "equity/us/large_cap/tech/AAPL")
        portfolio_weight : float, optional
            Portfolio weight (will be converted to relative weight if normalize_to_relative=True)
        benchmark_weight : float, optional  
            Benchmark weight (will be converted to relative weight if normalize_to_relative=True)
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
        PortfolioBuilderMultiplicative
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
        """Create parent-child edges for the hierarchical path - ONLY for explicitly added components."""
        # Only create edges between explicitly added components
        full_path = self.delimiter.join(path_parts)
        
        # Find the deepest explicitly added parent
        for i in range(len(path_parts) - 1, 0, -1):
            parent_path = self.delimiter.join(path_parts[:i])
            if parent_path in self._components_info:
                # Create edge from this parent to the current component
                edge = (parent_path, full_path)
                if edge not in self._edges:
                    self._edges.append(edge)
                break
        
        # If no explicit parent found, connect to root if this isn't the root
        if full_path != self.root_id:
            potential_edges = [(p, c) for p, c in self._edges if c == full_path]
            if not potential_edges:
                edge = (self.root_id, full_path)
                if edge not in self._edges:
                    self._edges.append(edge)
    
    def clear(self) -> 'PortfolioBuilderMultiplicative':
        """
        Clear all accumulated component and edge information.
        
        Returns
        -------
        PortfolioBuilderMultiplicative
            Self for method chaining
        """
        self._components_info.clear()
        self._edges.clear()
        return self
    
    def add_data(self, 
                 path: str, 
                 data: Dict[str, Any], 
                 create_if_missing: bool = False) -> 'PortfolioBuilderMultiplicative':
        """
        Add or update data for a component in the portfolio hierarchy.
        
        This helper function allows you to assign additional data to components
        that have already been added to the builder, or optionally create new
        components with data.
        
        Parameters
        ----------
        path : str
            Hierarchical path of the component (e.g., "equity/us/tech/AAPL")
        data : dict
            Dictionary of data to assign/merge with the component.
            New keys are added, existing keys are updated.
        create_if_missing : bool, default False
            If True, creates the component if it doesn't exist using add_path()
            If False, raises ValueError if component doesn't exist
            
        Returns
        -------
        PortfolioBuilderMultiplicative
            Self for method chaining
            
        Raises
        ------
        ValueError
            If path doesn't exist and create_if_missing=False
            
        Examples
        --------
        >>> builder = PortfolioBuilderMultiplicative()
        >>> builder.add_path("equity/us/tech/AAPL", portfolio_weight=0.05)
        >>> builder.add_data("equity/us/tech/AAPL", {
        ...     "sector": "Technology",
        ...     "market_cap": 3000000000,
        ...     "returns": [0.01, 0.02, -0.01, 0.03]
        ... })
        >>> 
        >>> # Chain multiple data additions
        >>> builder.add_data("equity/us/tech/AAPL", {"pe_ratio": 28.5}) \\
        ...        .add_data("equity/us/tech/MSFT", {"pe_ratio": 30.2})
        """
        # Validate that data is provided
        if not isinstance(data, dict) or not data:
            raise ValueError("Data must be a non-empty dictionary")
        
        # Check if component exists
        if path not in self._components_info:
            if create_if_missing:
                # Create the component with minimal parameters
                self.add_path(path=path, component_type='auto')
            else:
                raise ValueError(f"Component '{path}' not found. Use create_if_missing=True to create it.")
        
        # Get existing component info
        comp_info = self._components_info[path]
        
        # Merge data with existing data (new keys added, existing keys updated)
        if comp_info['data'] is None:
            comp_info['data'] = {}
        comp_info['data'].update(data)
        
        return self
        
    def from_paths(self, rows: List[Dict[str, Any]]) -> 'PortfolioBuilderMultiplicative':
        """
        Add multiple components from a list of path-based dictionaries.
        
        Parameters
        ----------
        rows : list of dict
            List of dictionaries containing path and weight information
            
        Returns
        -------
        PortfolioBuilderMultiplicative
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
    
    def from_hierarchy(self, hierarchy: Dict[str, Any], base_path: str = "") -> 'PortfolioBuilderMultiplicative':
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
        PortfolioBuilderMultiplicative
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
        Build the PortfolioGraph using multiplicative weight aggregation.
        
        Returns
        -------
        PortfolioGraph
            Constructed portfolio graph with relative weights for multiplication
        """
        # Add root component if not already added
        if self.root_id not in self._components_info:
            self._components_info[self.root_id] = {
                'id': self.root_id,
                'name': self.root_id,
                'path': self.root_id,
                'path_parts': [self.root_id],
                'portfolio_weight': 1.0,
                'benchmark_weight': 1.0,
                'component_type': 'node',
                'is_overlay': False,
                'data': {}
            }
            
        # Simple 4-step process
        self._validate_essentials()
        
        # Step 1: Ensure root unity FIRST (before any renormalization)
        if self.proportional_renormalize or self.normalize_to_relative:
            self._ensure_root_unity()
        
        # Step 2: Proportional renormalization if enabled
        if self.proportional_renormalize:
            self._proportional_renormalize()
        
        # Step 3: Convert to relative weights if enabled
        if self.normalize_to_relative:
            self._convert_to_relative_weights()
            
        return self._create_graph()
    
    def _validate_essentials(self) -> None:
        """Essential validation only - no over-engineering."""
        # Check shorts if not allowed
        if not self.allow_shorts:
            for comp_info in self._components_info.values():
                for weight_type in ['portfolio_weight', 'benchmark_weight']:
                    weight = comp_info.get(weight_type)
                    if weight is not None and weight < 0:
                        raise ValueError(
                            f"Negative weight not allowed: {comp_info['path']} "
                            f"{weight_type}={weight}"
                        )
    
    def _convert_to_relative_weights(self) -> None:
        """
        Single-phase conversion from absolute to relative weights.
        
        Key insight: We need to preserve original absolute weights during conversion
        to calculate proper relative weights (child_absolute / parent_absolute).
        """
        # Store original absolute weights before any conversion
        original_weights = {}
        for comp_id, comp_info in self._components_info.items():
            original_weights[comp_id] = {
                'portfolio_weight': comp_info.get('portfolio_weight'),
                'benchmark_weight': comp_info.get('benchmark_weight')
            }
        
        # Step 1: Top-down conversion using original absolute weights
        topo_order = self._get_topological_order()
        for component_id in topo_order:  # Start from root
            self._convert_children_to_relative(component_id, original_weights)
    
    def _get_topological_order(self) -> List[str]:
        """Get topological ordering of components for weight propagation."""
        # Build adjacency map
        adj_map = {}
        in_degree = {}
        
        # Initialize
        for comp_id in self._components_info:
            adj_map[comp_id] = []
            in_degree[comp_id] = 0
        
        # Build adjacency list and calculate in-degrees
        for parent_id, child_id in self._edges:
            adj_map[parent_id].append(child_id)
            in_degree[child_id] += 1
        
        # Kahn's algorithm for topological sorting
        queue = [comp_id for comp_id in self._components_info if in_degree[comp_id] == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in adj_map[current]:
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
    
    def _get_direct_children(self, component_id: str) -> List[str]:
        """Get direct children of a component."""
        children = []
        for parent_id, child_id in self._edges:
            if parent_id == component_id:
                children.append(child_id)
        return children
    
    def _calculate_parent_weight_from_children(self, parent_id: str) -> None:
        """Calculate parent weight as sum of core (non-overlay) children - ONLY for explicitly added components."""
        children = self._get_direct_children(parent_id)
        if not children:
            return
            
        # Only process if parent was explicitly added and has explicit weight
        if parent_id not in self._components_info:
            return
            
        parent_info = self._components_info[parent_id]
        
        # Skip if parent already has an explicitly set weight
        for weight_side in ['portfolio_weight', 'benchmark_weight']:
            if parent_info.get(weight_side) is not None:
                continue  # Keep explicitly set weight
                
            # Only calculate from children if no explicit weight was provided
            core_sum = 0.0
            has_weights = False
            
            for child_id in children:
                if child_id not in self._components_info:
                    continue
                    
                child_info = self._components_info[child_id]
                child_weight = child_info.get(weight_side)
                
                # Include in core sum (exclude overlays)
                if not child_info.get('is_overlay', False) and child_weight is not None:
                    core_sum += child_weight
                    has_weights = True
            
            # Set parent weight to core children sum only if no explicit weight
            if has_weights:
                parent_info[weight_side] = core_sum
    
    def _ensure_root_unity(self) -> None:
        """Ensure root weight equals 1.0 for multiplicative consistency."""
        # Find root component (component with no parents)
        root_components = []
        for comp_id in self._components_info:
            is_root = True
            for parent_id, child_id in self._edges:
                if child_id == comp_id:
                    is_root = False
                    break
            if is_root:
                root_components.append(comp_id)
        
        # Use specified root or first root found
        if self.root_id in root_components:
            primary_root = self.root_id
        elif root_components:
            primary_root = root_components[0]
        else:
            return  # No root found
        
        # Set root weight to 1.0
        if primary_root in self._components_info:
            self._components_info[primary_root]['portfolio_weight'] = 1.0
            self._components_info[primary_root]['benchmark_weight'] = 1.0
    
    def _convert_children_to_relative(self, parent_id: str, original_weights: dict) -> None:
        """Convert children to relative weights using the specified reconciliation policy."""
        children = self._get_direct_children(parent_id)
        if not children:
            return
        
        for weight_side in ['portfolio_weight', 'benchmark_weight']:
            # Collect valid children and their weights
            children_sum = 0.0
            valid_children = []
            
            for child_id in children:
                if child_id not in self._components_info:
                    continue
                    
                original_child = original_weights.get(child_id, {})
                child_absolute = original_child.get(weight_side)
                child_info = self._components_info[child_id]
                
                if child_absolute is not None and not child_info.get('is_overlay', False):
                    children_sum += child_absolute
                    valid_children.append((child_id, child_absolute))
            
            if children_sum == 0 or not valid_children:
                continue
                
            # Apply reconciliation policy
            if self.weight_reconcile_policy == 'respect_parent':
                # Scale children proportionally to fit parent weight, then convert to relative
                original_parent = original_weights.get(parent_id, {})
                target_parent_weight = original_parent.get(weight_side)
                
                if target_parent_weight is not None and target_parent_weight != 0:
                    # Calculate scaling factor to fit parent weight
                    scale_factor = target_parent_weight / children_sum
                    
                    # Scale each child and convert to relative share of target parent
                    for child_id, child_absolute in valid_children:
                        scaled_child_weight = child_absolute * scale_factor
                        child_relative = scaled_child_weight / target_parent_weight
                        self._components_info[child_id][weight_side] = child_relative
                    
                    # Update parent's absolute weight to the target (for next level normalization)
                    self._components_info[parent_id][weight_side] = target_parent_weight
                    
                else:
                    # No parent weight specified, fallback to respect_children behavior
                    for child_id, child_absolute in valid_children:
                        child_relative = child_absolute / children_sum
                        self._components_info[child_id][weight_side] = child_relative
            else:
                # 'respect_children': normalize children among themselves (current behavior)
                for child_id, child_absolute in valid_children:
                    child_relative = child_absolute / children_sum
                    self._components_info[child_id][weight_side] = child_relative
                    
            # Overlays keep their absolute weights for operational tracking
    
    def _proportional_renormalize(self) -> None:
        """
        Proportionally renormalize weights: normalize leaf weights and propagate up hierarchy.
        
        Process:
        1. Identify leaf components (no children)
        2. Normalize non-overlay leaf weights: weights / sum(weights) 
        3. Propagate weights bottom-up to calculate node weights
        4. Overlay components preserve their original weights
        """
        # Step 1: Identify leaf components
        leaf_components = []
        for comp_id in self._components_info:
            if self._is_leaf_component(comp_id):
                leaf_components.append(comp_id)
        
        # Step 2: Normalize leaf weights for each weight type
        for weight_side in ['portfolio_weight', 'benchmark_weight']:
            self._normalize_leaf_weights(leaf_components, weight_side)
            
        # Step 3: Propagate weights bottom-up through hierarchy
        self._propagate_weights_bottom_up()
    
    def _normalize_leaf_weights(self, leaf_components: List[str], weight_side: str) -> None:
        """Normalize leaf weights proportionally, excluding overlays."""
        # Collect non-overlay leaf weights
        leaf_weights = []
        non_overlay_leaves = []
        
        for comp_id in leaf_components:
            comp_info = self._components_info[comp_id]
            weight = comp_info.get(weight_side)
            is_overlay = comp_info.get('is_overlay', False)
            
            if weight is not None and not is_overlay:
                leaf_weights.append(weight)
                non_overlay_leaves.append(comp_id)
        
        if not leaf_weights or sum(leaf_weights) == 0:
            return  # No weights to normalize
            
        # Calculate normalized weights: weights / sum(weights)
        total_weight = sum(leaf_weights)
        
        for i, comp_id in enumerate(non_overlay_leaves):
            normalized_weight = leaf_weights[i] / total_weight
            self._components_info[comp_id][weight_side] = normalized_weight
    
    def _propagate_weights_bottom_up(self) -> None:
        """Propagate weights from leaves up to root using bottom-up traversal."""
        # Get reverse topological order (bottom-up)
        topo_order = self._get_topological_order()
        reverse_topo = list(reversed(topo_order))
        
        for weight_side in ['portfolio_weight', 'benchmark_weight']:
            for comp_id in reverse_topo:
                if not self._is_leaf_component(comp_id):
                    # This is a node - calculate weight from children
                    self._calculate_node_weight_from_children(comp_id, weight_side)
    
    def _calculate_node_weight_from_children(self, node_id: str, weight_side: str) -> None:
        """Calculate node weight as sum of non-overlay children weights."""
        children = self._get_direct_children(node_id)
        if not children:
            return
            
        total_weight = 0.0
        has_children_with_weights = False
        
        for child_id in children:
            if child_id not in self._components_info:
                continue
                
            child_info = self._components_info[child_id]
            child_weight = child_info.get(weight_side)
            is_overlay = child_info.get('is_overlay', False)
            
            # Sum non-overlay children weights
            if child_weight is not None and not is_overlay:
                total_weight += child_weight
                has_children_with_weights = True
        
        # Set the calculated weight for the node
        if has_children_with_weights:
            self._components_info[node_id][weight_side] = total_weight

    def _store_data_as_metrics(self, graph: 'PortfolioGraph', component_id: str, data: dict) -> None:
        """Store data dict entries as metrics in the metric store."""
        if not data:
            return
            
        from .metrics import ScalarMetric, SeriesMetric
        import pandas as pd
        
        for key, value in data.items():
            try:
                # Determine metric type based on value
                if isinstance(value, pd.Series):
                    # Already a pandas Series - use directly
                    metric = SeriesMetric(value)
                    graph.metric_store.set_metric(component_id, key, metric)
                elif isinstance(value, (list, tuple)):
                    # Time series data (e.g., portfolio returns)
                    if len(value) > 0:
                        # Convert to pandas Series for SeriesMetric
                        series = pd.Series(value, name=key)
                        metric = SeriesMetric(series)
                        graph.metric_store.set_metric(component_id, key, metric)
                elif isinstance(value, (int, float)):
                    # Scalar data
                    metric = ScalarMetric(float(value))
                    graph.metric_store.set_metric(component_id, key, metric)
                elif isinstance(value, str):
                    # String data - store as object metric for now
                    # Skip strings as metrics are typically numeric
                    continue
                else:
                    # Skip complex objects for now
                    continue
                    
            except Exception:
                # If metric creation fails, silently skip to avoid breaking the build
                continue
    
    def _create_graph(self) -> 'PortfolioGraph':
        """Create the PortfolioGraph from processed component data."""
        # Import here to avoid circular imports
        from .graph import PortfolioGraph
        from .components import PortfolioNode, PortfolioLeaf
        from .metrics import ScalarMetric
        
        # Create the graph
        graph = PortfolioGraph(root_id=self.root_id, metric_store=self.metric_store)
        
        # Add all components to the graph (including root from _components_info)
        for component_id, comp_info in self._components_info.items():
                
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
            
            # Store weights (now relative after normalization)
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
            
            # Store additional metrics from data dict
            self._store_data_as_metrics(graph, component_id, comp_info['data'])
        
        # Create edges and connect to root if needed
        self._create_graph_edges(graph)
        
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