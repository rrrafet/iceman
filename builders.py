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
    from .metrics import MetricStore, ValidationIssue

from .metrics import InMemoryMetricStore, ValidationIssue, WeightPathAggregator, WeightPathAggregatorSum


class PortfolioBuilder:
    """
    Builder class for constructing portfolio + benchmark hierarchies with enhanced features.
    
    Supports:
    - Hierarchical dict and path-based row inputs
    - Portfolio + benchmark weights with relative/absolute types
    - Leverage and short positions (weights need not sum to 1.0)
    - Multiple reconciliation policies and auto-balance modes
    - Dual aggregation strategies (product/sum) with auto-selection
    - Comprehensive validation and export capabilities
    """
    
    def __init__(self,
                 *,
                 normalize: bool = True,
                 allow_shorts: bool = False,
                 enforce_sum_to_one: bool = False,
                 target_net_exposure: Optional[float] = None,
                 max_gross_exposure: Optional[float] = None,
                 auto_balance: Literal["none", "synthetic_financing", "normalize"] = "none",
                 financing_component_id: str = "__financing__",
                 reconcile_policy: Literal["respect_parent", "respect_children", "error"] = "respect_parent",
                 reconcile_policy_benchmark: Optional[Literal["respect_parent", "respect_children", "error"]] = None,
                 propagation_strategy: Literal["auto", "product", "sum"] = "auto",
                 auto_normalize_hierarchy: bool = True,
                 overlay_weight_mode: Literal["dual", "allocation_only"] = "dual",
                 short_aggregation_mode: Literal["include", "separate"] = "include",
                 tol: float = 1e-8,
                 delimiter: str = "/",
                 root_id: str = 'portfolio',
                 metric_store: Optional['MetricStore'] = None):
        """
        Initialize the enhanced portfolio builder.
        
        Parameters
        ----------
        normalize : bool, default True
            Whether to normalize weights (only used when normalization is explicitly enabled)
        allow_shorts : bool, default False
            Whether to allow negative weights
        enforce_sum_to_one : bool, default False
            If True, force per-parent relative sums to 1
        target_net_exposure : float, optional
            Target net exposure (e.g., 1.0). Used with auto_balance modes
        max_gross_exposure : float, optional
            Optional maximum gross exposure guard
        auto_balance : {"none", "synthetic_financing", "normalize"}, default "none"
            Auto-balance mode for target net exposure
        financing_component_id : str, default "__financing__"
            ID for synthetic financing component
        reconcile_policy : {"respect_parent", "respect_children", "error"}, default "respect_parent"
            Policy for reconciling weight mismatches
        reconcile_policy_benchmark : {"respect_parent", "respect_children", "error"}, optional
            Override reconcile_policy for benchmark side
        propagation_strategy : {"auto", "product", "sum"}, default "auto"
            Weight aggregation strategy selection
        auto_normalize_hierarchy : bool, default True
            Enable automatic weight normalization to ensure WeightPathAggregator consistency
        overlay_weight_mode : {"dual", "allocation_only"}, default "dual"
            How to handle overlay position weights: dual tracking or allocation only
        short_aggregation_mode : {"include", "separate"}, default "include"
            Whether to include short positions in parent weight aggregation
        tol : float, default 1e-8
            Numerical tolerance for validations
        delimiter : str, default "/"
            Path delimiter character
        root_id : str, default 'portfolio'
            ID for the root component
        metric_store : MetricStore, optional
            Metric store to use. If None, creates new InMemoryMetricStore
        """
        # Configuration parameters
        self.normalize = normalize
        self.allow_shorts = allow_shorts
        self.enforce_sum_to_one = enforce_sum_to_one
        self.target_net_exposure = target_net_exposure
        self.max_gross_exposure = max_gross_exposure
        self.auto_balance = auto_balance
        self.financing_component_id = financing_component_id
        self.reconcile_policy = reconcile_policy
        self.reconcile_policy_benchmark = reconcile_policy_benchmark or reconcile_policy
        self.propagation_strategy = propagation_strategy
        self.auto_normalize_hierarchy = auto_normalize_hierarchy
        self.overlay_weight_mode = overlay_weight_mode
        self.short_aggregation_mode = short_aggregation_mode
        self.tol = tol
        self.delimiter = delimiter
        self.root_id = root_id
        self.metric_store = metric_store or InMemoryMetricStore()
        
        # Internal storage
        self._components_info = {}  # Store component info before building graph
        self._edges = []  # Store edge relationships
        self._weight_hierarchy = {}  # Store hierarchical weight structure
        self._templates = {}  # Store reusable templates
        
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
        self._clear_internal_state()
        return self
    
    def add_relative_weights(self, weight_hierarchy: Dict[str, Any]) -> 'PortfolioBuilder':
        """
        Add hierarchical weight structure with relative weights that auto-normalize.
        
        Parameters
        ----------
        weight_hierarchy : dict
            Nested dictionary defining relative weights:
            {
                "equity": {
                    "portfolio_weight": 0.7,
                    "benchmark_weight": 0.6,
                    "children": {
                        "us": {"relative_portfolio": 0.65, "relative_benchmark": 0.70},
                        "international": {"relative_portfolio": 0.35, "relative_benchmark": 0.30}
                    }
                }
            }
            
        Returns
        -------
        PortfolioBuilder
            Self for method chaining
        """
        self._weight_hierarchy.update(weight_hierarchy)
        return self
    
    def add_template(self, name: str, template: Dict[str, Any]) -> 'PortfolioBuilder':
        """
        Add a reusable template for hierarchy construction.
        
        Parameters
        ----------
        name : str
            Template name
        template : dict
            Template structure defining hierarchy pattern
            
        Returns
        -------
        PortfolioBuilder
            Self for method chaining
        """
        self._templates[name] = template
        return self
    
    def apply_template(self, 
                      base_path: str, 
                      template_name: str,
                      scale_portfolio: float = 1.0,
                      scale_benchmark: float = 1.0) -> 'PortfolioBuilder':
        """
        Apply a template to create hierarchy structure under a base path.
        
        Parameters
        ----------
        base_path : str
            Base path to apply template under
        template_name : str
            Name of template to apply
        scale_portfolio : float, default 1.0
            Scaling factor for portfolio weights
        scale_benchmark : float, default 1.0
            Scaling factor for benchmark weights
            
        Returns
        -------
        PortfolioBuilder
            Self for method chaining
        """
        if template_name not in self._templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self._templates[template_name]
        self._apply_template_recursive(base_path, template, scale_portfolio, scale_benchmark)
        return self
    
    def _apply_template_recursive(self, 
                                 base_path: str, 
                                 template: Dict[str, Any],
                                 scale_portfolio: float,
                                 scale_benchmark: float):
        """Recursively apply template structure."""
        for key, value in template.items():
            current_path = f"{base_path}/{key}" if base_path else key
            
            if isinstance(value, dict):
                if 'children' in value:
                    # This is a node with children
                    port_weight = value.get('portfolio_weight', 0.0) * scale_portfolio
                    bench_weight = value.get('benchmark_weight', 0.0) * scale_benchmark
                    
                    self.add_path(current_path, 
                                portfolio_weight=port_weight,
                                benchmark_weight=bench_weight,
                                component_type='node')
                    
                    # Recursively apply children
                    self._apply_template_recursive(current_path, value['children'], 
                                                 scale_portfolio, scale_benchmark)
                else:
                    # This is a leaf with weight value
                    if isinstance(value, (int, float)):
                        self.add_path(current_path,
                                    portfolio_weight=value * scale_portfolio,
                                    benchmark_weight=value * scale_benchmark,
                                    component_type='leaf')
                    else:
                        # Value is a dict with portfolio/benchmark weights
                        port_weight = value.get('portfolio_weight', 0.0) * scale_portfolio
                        bench_weight = value.get('benchmark_weight', 0.0) * scale_benchmark
                        
                        self.add_path(current_path,
                                    portfolio_weight=port_weight,
                                    benchmark_weight=bench_weight,
                                    component_type='leaf')
    
    def _normalize_relative_weights(self):
        """Normalize relative weights to absolute weights."""
        def process_hierarchy(hierarchy_dict, parent_portfolio_weight=1.0, parent_benchmark_weight=1.0, base_path=""):
            for key, value in hierarchy_dict.items():
                current_path = f"{base_path}/{key}" if base_path else key
                
                # Get absolute weights for this component
                port_weight = value.get('portfolio_weight', parent_portfolio_weight)
                bench_weight = value.get('benchmark_weight', parent_benchmark_weight)
                
                # Create or update component info
                if current_path not in self._components_info:
                    # Create component if it doesn't exist
                    child_parts = current_path.split('/')
                    self._components_info[current_path] = {
                        'id': current_path,
                        'name': key,
                        'type': 'node' if 'children' in value else 'leaf',
                        'portfolio_weight': port_weight,
                        'benchmark_weight': bench_weight,
                        'data': None,
                        'path_parts': child_parts
                    }
                    
                    # Add edge to parent if not root
                    if base_path:
                        self._edges.append((base_path, current_path))
                else:
                    # Update existing component weights
                    self._components_info[current_path]['portfolio_weight'] = port_weight
                    self._components_info[current_path]['benchmark_weight'] = bench_weight
                
                # Process children if they exist
                if 'children' in value:
                    for child_key, child_value in value['children'].items():
                        child_path = f"{current_path}/{child_key}"
                        
                        # Calculate absolute weights from relative weights
                        rel_port = child_value.get('relative_portfolio', 1.0)
                        rel_bench = child_value.get('relative_benchmark', 1.0)
                        
                        abs_port = port_weight * rel_port
                        abs_bench = bench_weight * rel_bench
                        
                        # Create child_value copy with absolute weights for recursion
                        child_value_copy = child_value.copy()
                        child_value_copy['portfolio_weight'] = abs_port
                        child_value_copy['benchmark_weight'] = abs_bench
                        
                        # Recursively process this child
                        process_hierarchy({child_key: child_value_copy}, abs_port, abs_bench, current_path)
        
        if self._weight_hierarchy:
            process_hierarchy(self._weight_hierarchy)
    
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
        
        # Validate and reconcile weights first (always)
        self._validate_and_reconcile_weights()
        
        # Auto-normalize weights after reconciliation (if enabled)
        self._auto_normalize_weights()
        
        # Normalize relative weights first
        self._normalize_relative_weights()
        
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
    
    def validate_weights(self) -> Tuple[bool, List[str]]:
        """
        Validate weight consistency across the hierarchy.
        
        Returns
        -------
        tuple
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Group components by parent
        children_by_parent = {}
        for parent_id, child_id in self._edges:
            if parent_id not in children_by_parent:
                children_by_parent[parent_id] = []
            children_by_parent[parent_id].append(child_id)
        
        # Check weight consistency for each parent
        for parent_id, child_ids in children_by_parent.items():
            if parent_id not in self._components_info:
                continue
                
            parent_info = self._components_info[parent_id]
            parent_port_weight = parent_info.get('portfolio_weight', 0.0)
            parent_bench_weight = parent_info.get('benchmark_weight', 0.0)
            
            # Sum child weights
            child_port_sum = 0.0
            child_bench_sum = 0.0
            
            for child_id in child_ids:
                if child_id in self._components_info:
                    child_info = self._components_info[child_id]
                    child_port_weight = child_info.get('portfolio_weight')
                    child_bench_weight = child_info.get('benchmark_weight')
                    
                    if child_port_weight is not None:
                        child_port_sum += child_port_weight
                    if child_bench_weight is not None:
                        child_bench_sum += child_bench_weight
            
            # Check consistency
            if parent_port_weight and abs(child_port_sum - parent_port_weight) > 1e-6:
                issues.append(f"Portfolio weight mismatch for {parent_id}: "
                            f"parent={parent_port_weight:.6f}, children_sum={child_port_sum:.6f}")
            
            if parent_bench_weight and abs(child_bench_sum - parent_bench_weight) > 1e-6:
                issues.append(f"Benchmark weight mismatch for {parent_id}: "
                            f"parent={parent_bench_weight:.6f}, children_sum={child_bench_sum:.6f}")
        
        return len(issues) == 0, issues
    
    def validate(self, graph: "PortfolioGraph") -> List["ValidationIssue"]:
        """
        Validate portfolio graph for consistency and constraints.
        
        Parameters
        ----------
        graph : PortfolioGraph
            Graph to validate
            
        Returns
        -------
        list of ValidationIssue
            List of validation issues found
        """
        issues = []
        
        # Basic structure validation
        issues.extend(self._validate_structure(graph))
        
        # Weight consistency validation
        issues.extend(self._validate_weights(graph))
        
        # Exposure limits validation
        issues.extend(self._validate_exposures(graph))
        
        # Leverage/shorts validation
        issues.extend(self._validate_leverage_shorts(graph))
        
        return issues
    
    def to_paths(self, graph: "PortfolioGraph") -> List[Dict[str, Any]]:
        """
        Export PortfolioGraph to path-based row format.
        
        Parameters
        ----------
        graph : PortfolioGraph
            Graph to export
            
        Returns
        -------
        list of dict
            Path-based row data
        """
        rows = []
        
        for component_id, component in graph.components.items():
            # Build path by traversing parents
            path = self._build_component_path(component_id, graph)
            
            # Get weights
            portfolio_weight = self._get_component_weight(component, 'portfolio_weight')
            benchmark_weight = self._get_component_weight(component, 'benchmark_weight')
            
            # Create row
            row = {
                'type': 'leaf' if component.is_leaf() else 'node',
                'path': path,
                'component_id': component_id,
                'portfolio_weight': portfolio_weight,
                'portfolio_weight_type': 'absolute',  # Default for export
                'benchmark_weight': benchmark_weight,
                'benchmark_weight_type': 'absolute',  # Default for export
                'meta': getattr(component, 'data', None)
            }
            rows.append(row)
            
        return rows
    
    def to_hierarchy(self, graph: "PortfolioGraph") -> Dict[str, Any]:
        """
        Export PortfolioGraph to hierarchical dictionary format.
        
        Parameters
        ----------
        graph : PortfolioGraph
            Graph to export
            
        Returns
        -------
        dict
            Hierarchical dictionary structure
        """
        # Find root component
        root_component = None
        for component in graph.components.values():
            if len(component.parent_ids) == 0:
                root_component = component
                break
        
        if root_component is None:
            raise ValueError("No root component found in graph")
        
        return self._build_hierarchy_node(root_component, graph)
    
    def _clear_internal_state(self) -> None:
        """Clear all internal state for a fresh build."""
        self._components_info.clear()
        self._edges.clear()
        self._weight_hierarchy.clear()
    
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
    
    def _validate_and_reconcile_weights(self) -> None:
        """Validate and reconcile weights according to policies."""
        # Apply reconciliation policies
        self._reconcile_weight_mismatches()
        
        # Apply auto-balance if configured
        if self.auto_balance != "none":
            self._apply_auto_balance()
        
        # Validate final weights
        self._validate_final_weights()
    
    def _reconcile_weight_mismatches(self) -> None:
        """Reconcile weight mismatches according to reconcile policies."""
        # Group components by parent to check for weight consistency
        parent_children = {}
        for parent_path, child_path in self._edges:
            if parent_path not in parent_children:
                parent_children[parent_path] = []
            parent_children[parent_path].append(child_path)
        
        for parent_path, child_paths in parent_children.items():
            if parent_path not in self._components_info:
                continue
                
            parent_info = self._components_info[parent_path]
            
            # Check both portfolio and benchmark sides
            for weight_side in ['portfolio', 'benchmark']:
                policy = self.reconcile_policy_benchmark if weight_side == 'benchmark' else self.reconcile_policy
                self._reconcile_weight_side(parent_path, child_paths, weight_side, policy)
    
    def _reconcile_weight_side(self, parent_path: str, child_paths: List[str], weight_side: str, policy: str) -> None:
        """Reconcile weights for a specific side (portfolio or benchmark)."""
        parent_info = self._components_info[parent_path]
        weight_key = f'{weight_side}_weight'
        weight_type_key = f'{weight_side}_weight_type'
        
        parent_weight = parent_info.get(weight_key)
        parent_weight_type = parent_info.get(weight_type_key, 'absolute')
        
        # Get children weights
        children_weights = []
        children_weight_types = []
        
        for child_path in child_paths:
            if child_path in self._components_info:
                child_info = self._components_info[child_path]
                children_weights.append(child_info.get(weight_key, 0.0))
                children_weight_types.append(child_info.get(weight_type_key, 'absolute'))
        
        if not children_weights:
            return
        
        # Handle edge case: all children weights are None or zero
        all_zero_or_none = all(w is None or w == 0.0 for w in children_weights)
        if all_zero_or_none and not self.enforce_sum_to_one:
            return  # Skip if not enforcing constraints
        elif all_zero_or_none and self.enforce_sum_to_one and all(wt == 'relative' for wt in children_weight_types):
            # Special case: enforce_sum_to_one with all zero relative weights
            raise ValueError(
                f"Relative weights for {parent_path} ({weight_side}) sum to 0.0, "
                f"which violates enforce_sum_to_one constraint"
            )
        
        # Handle relative vs absolute weight reconciliation
        if parent_weight_type == 'absolute' and all(wt == 'relative' for wt in children_weight_types):
            # Parent absolute, children relative: derive child absolutes
            if parent_weight is not None:
                for i, child_path in enumerate(child_paths):
                    if child_path in self._components_info:
                        child_rel_weight = children_weights[i]
                        child_abs_weight = parent_weight * child_rel_weight
                        self._components_info[child_path][weight_key] = child_abs_weight
                        self._components_info[child_path][weight_type_key] = 'absolute'
        
        elif parent_weight_type == 'relative' and all(wt == 'absolute' for wt in children_weight_types):
            # Children absolute, parent relative: this shouldn't happen at root level, handle carefully
            pass
        
        elif parent_weight_type == 'absolute' and all(wt == 'absolute' for wt in children_weight_types):
            # Both absolute: check for mismatches and reconcile (exclude overlays from allocation calculations)
            # Calculate sum only for core (non-overlay) children
            core_children_sum = 0.0
            core_child_indices = []
            
            for i, child_path in enumerate(child_paths):
                if child_path in self._components_info:
                    child_info = self._components_info[child_path]
                    if not child_info.get('is_overlay', False) and children_weights[i] is not None:
                        core_children_sum += children_weights[i]
                        core_child_indices.append(i)
            
            if parent_weight is not None and abs(core_children_sum - parent_weight) > self.tol:
                # Mismatch detected, apply reconciliation policy (only to core children)
                if policy == "respect_parent":
                    # Scale core children to match parent (overlays unchanged)
                    if core_children_sum > 0:
                        scale_factor = parent_weight / core_children_sum
                        for i in core_child_indices:
                            child_path = child_paths[i]
                            if child_path in self._components_info and children_weights[i] is not None:
                                self._components_info[child_path][weight_key] = children_weights[i] * scale_factor
                
                elif policy == "respect_children":
                    # Update parent to match core children sum (excluding overlays)
                    self._components_info[parent_path][weight_key] = core_children_sum
                
                elif policy == "error":
                    # Raise error for mismatch
                    raise ValueError(
                        f"Weight mismatch for {parent_path} ({weight_side}): "
                        f"parent={parent_weight}, core_children_sum={core_children_sum}"
                    )
        
        # Handle relative weight normalization - exclude overlay strategies
        if all(wt == 'relative' for wt in children_weight_types):
            # Separate core and overlay children for normalization
            core_weights = []
            core_indices = []
            overlay_indices = []
            
            for i, child_path in enumerate(child_paths):
                if child_path in self._components_info:
                    child_info = self._components_info[child_path]
                    if child_info.get('is_overlay', False):
                        overlay_indices.append(i)
                    else:
                        core_weights.append(children_weights[i] if children_weights[i] is not None else 0.0)
                        core_indices.append(i)
            
            # Calculate normalization only for core (non-overlay) children
            rel_sum = sum(core_weights)
            
            # Handle edge case: zero or very small sum for core children
            if rel_sum <= self.tol:
                if self.enforce_sum_to_one:
                    raise ValueError(
                        f"Core (non-overlay) relative weights for {parent_path} ({weight_side}) sum to {rel_sum}, "
                        f"which is too small for normalization (tolerance: {self.tol})"
                    )
                return  # Cannot normalize zero weights
            
            # Check if normalization is needed for core children
            needs_normalization = abs(rel_sum - 1.0) > self.tol
            
            if needs_normalization:
                if self.normalize:
                    # Normalize only core children weights to sum to 1.0
                    for i in core_indices:
                        child_path = child_paths[i]
                        if child_path in self._components_info and children_weights[i] is not None:
                            normalized_weight = children_weights[i] / rel_sum
                            self._components_info[child_path][weight_key] = normalized_weight
                    
                    # Overlay children keep their allocation weight (0.0) unchanged
                    # Their operational weight will be handled separately by the context system
                    
                elif self.enforce_sum_to_one:
                    # Enforce sum to one without normalization: raise error
                    raise ValueError(
                        f"Core (non-overlay) relative weights for {parent_path} ({weight_side}) sum to {rel_sum}, "
                        f"not 1.0 (tolerance: {self.tol})"
                    )
    
    def _apply_auto_balance(self) -> None:
        """Apply auto-balance mode if configured."""
        if self.target_net_exposure is None:
            return
        
        # Apply auto-balance to root level components
        root_children = []
        for parent_path, child_path in self._edges:
            if parent_path == self.root_id or parent_path == "":
                root_children.append(child_path)
        
        if not root_children:
            return
        
        for weight_side in ['portfolio', 'benchmark']:
            self._apply_auto_balance_side(root_children, weight_side)
    
    def _apply_auto_balance_side(self, components: List[str], weight_side: str) -> None:
        """Apply auto-balance for a specific weight side."""
        weight_key = f'{weight_side}_weight'
        
        # Calculate current net exposure
        current_net = 0.0
        for comp_path in components:
            if comp_path in self._components_info:
                weight = self._components_info[comp_path].get(weight_key, 0.0)
                if weight is not None:
                    current_net += weight
        
        net_diff = self.target_net_exposure - current_net
        
        if abs(net_diff) <= self.tol:
            return  # Already at target
        
        if self.auto_balance == "synthetic_financing":
            # Add synthetic financing component
            financing_path = self.financing_component_id
            
            # Create financing component if it doesn't exist
            if financing_path not in self._components_info:
                self._components_info[financing_path] = {
                    'id': self.financing_component_id,
                    'name': 'Synthetic Financing',
                    'type': 'leaf',
                    'portfolio_weight': 0.0,
                    'portfolio_weight_type': 'absolute',
                    'benchmark_weight': 0.0,
                    'benchmark_weight_type': 'absolute',
                    'data': {'synthetic': True},
                    'path_parts': [financing_path]
                }
                
                # Add edge to root
                self._edges.append((self.root_id, financing_path))
            
            # Set financing weight to balance to target
            self._components_info[financing_path][weight_key] = net_diff
        
        elif self.auto_balance == "normalize":
            # Normalize existing components proportionally
            if current_net > 0 and self.target_net_exposure != 0:
                scale_factor = self.target_net_exposure / current_net
                
                for comp_path in components:
                    if comp_path in self._components_info:
                        current_weight = self._components_info[comp_path].get(weight_key, 0.0)
                        if current_weight is not None:
                            self._components_info[comp_path][weight_key] = current_weight * scale_factor
    
    def _validate_final_weights(self) -> None:
        """Validate final weights against constraints."""
        # Validate shorts are allowed if present
        if not self.allow_shorts:
            for comp_path, comp_info in self._components_info.items():
                for weight_side in ['portfolio', 'benchmark']:
                    weight = comp_info.get(f'{weight_side}_weight')
                    if weight is not None and weight < -self.tol:
                        raise ValueError(
                            f"Negative weight not allowed: {comp_path} {weight_side}={weight}"
                        )
        
        # Validate exposure limits if set
        if self.max_gross_exposure is not None:
            for weight_side in ['portfolio', 'benchmark']:
                gross_exposure = self._calculate_gross_exposure(weight_side)
                if gross_exposure > self.max_gross_exposure:
                    raise ValueError(
                        f"Gross exposure {gross_exposure:.4f} exceeds limit {self.max_gross_exposure:.4f} "
                        f"for {weight_side}"
                    )
    
    def _calculate_gross_exposure(self, weight_side: str) -> float:
        """Calculate gross exposure for a weight side."""
        weight_key = f'{weight_side}_weight'
        gross_exposure = 0.0
        
        for comp_info in self._components_info.values():
            weight = comp_info.get(weight_key, 0.0)
            if weight is not None:
                gross_exposure += abs(weight)
        
        return gross_exposure
    
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
                
                # Include in core sum based on configuration
                if child_info.get('is_overlay', False):
                    # Overlays don't contribute to parent allocation weight
                    continue
                elif child_weight is not None:
                    if self.short_aggregation_mode == "include":
                        # Include shorts in parent weight calculation
                        core_sum += child_weight
                    elif child_weight >= 0:
                        # Only include positive weights if separating shorts
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
    
    def _validate_structure(self, graph: "PortfolioGraph") -> List["ValidationIssue"]:
        """Validate basic graph structure."""
        issues = []
        
        # Check for duplicate component IDs
        component_ids = set()
        for component_id in graph.components:
            if component_id in component_ids:
                issues.append(ValidationIssue(
                    "duplicate_id",
                    f"Duplicate component ID: {component_id}",
                    path=component_id
                ))
            component_ids.add(component_id)
        
        # Check for missing parents
        for parent_id, children in graph._adjacency_list.items():
            if parent_id not in graph.components:
                issues.append(ValidationIssue(
                    "missing_parent",
                    f"Parent component not found: {parent_id}",
                    path=parent_id
                ))
        
        # Check for circular dependencies
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph._adjacency_list.get(node, []):
                if has_cycle(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for component_id in graph.components:
            if component_id not in visited:
                if has_cycle(component_id):
                    issues.append(ValidationIssue(
                        "circular_dependency",
                        f"Circular dependency detected involving: {component_id}",
                        path=component_id
                    ))
        
        return issues
    
    def _validate_weights(self, graph: "PortfolioGraph") -> List["ValidationIssue"]:
        """Validate weight consistency."""
        issues = []
        
        # Check for weight consistency between parent and children
        for parent_id, children in graph._adjacency_list.items():
            for weight_side in ['portfolio', 'benchmark']:
                weight_metric_name = f'{weight_side}_weight'
                
                parent_metric = graph.metric_store.get_metric(parent_id, weight_metric_name)
                parent_weight = parent_metric.value() if parent_metric else None
                
                children_sum = 0.0
                for child_id in children:
                    child_metric = graph.metric_store.get_metric(child_id, weight_metric_name)
                    if child_metric:
                        children_sum += child_metric.value()
                
                if parent_weight is not None and abs(children_sum - parent_weight) > self.tol:
                    issues.append(ValidationIssue(
                        "weight_mismatch",
                        f"Weight mismatch: parent={parent_weight:.6f}, children_sum={children_sum:.6f}",
                        path=parent_id,
                        side=weight_side,
                        severity="warning"
                    ))
        
        return issues
    
    def _validate_exposures(self, graph: "PortfolioGraph") -> List["ValidationIssue"]:
        """Validate exposure limits."""
        issues = []
        
        if self.max_gross_exposure is not None:
            for weight_side in ['portfolio', 'benchmark']:
                weight_metric_name = f'{weight_side}_weight'
                gross_exposure = 0.0
                
                for component_id in graph.components:
                    weight_metric = graph.metric_store.get_metric(component_id, weight_metric_name)
                    if weight_metric:
                        gross_exposure += abs(weight_metric.value())
                
                if gross_exposure > self.max_gross_exposure:
                    issues.append(ValidationIssue(
                        "gross_exposure_exceeded",
                        f"Gross exposure {gross_exposure:.4f} exceeds limit {self.max_gross_exposure:.4f}",
                        side=weight_side
                    ))
        
        if self.target_net_exposure is not None:
            for weight_side in ['portfolio', 'benchmark']:
                weight_metric_name = f'{weight_side}_weight'
                net_exposure = 0.0
                
                for component_id in graph.components:
                    weight_metric = graph.metric_store.get_metric(component_id, weight_metric_name)
                    if weight_metric:
                        net_exposure += weight_metric.value()
                
                if abs(net_exposure - self.target_net_exposure) > self.tol:
                    issues.append(ValidationIssue(
                        "net_exposure_mismatch",
                        f"Net exposure {net_exposure:.4f} differs from target {self.target_net_exposure:.4f}",
                        side=weight_side,
                        severity="warning"
                    ))
        
        return issues
    
    def _validate_leverage_shorts(self, graph: "PortfolioGraph") -> List["ValidationIssue"]:
        """Validate leverage and shorts constraints."""
        issues = []
        
        if not self.allow_shorts:
            for component_id in graph.components:
                for weight_side in ['portfolio', 'benchmark']:
                    weight_metric_name = f'{weight_side}_weight'
                    weight_metric = graph.metric_store.get_metric(component_id, weight_metric_name)
                    
                    if weight_metric and weight_metric.value() < -self.tol:
                        issues.append(ValidationIssue(
                            "negative_weight_not_allowed",
                            f"Negative weight detected: {weight_metric.value():.6f}",
                            path=component_id,
                            side=weight_side
                        ))
        
        return issues
    
    def _build_component_path(self, component_id: str, graph: "PortfolioGraph") -> str:
        """Build full path for a component."""
        # Path building logic would go here
        return component_id  # Placeholder
    
    def _get_component_weight(self, component, weight_type: str) -> Optional[float]:
        """Get weight for a component."""
        # Weight extraction logic would go here
        return None  # Placeholder
    
    def _build_hierarchy_node(self, component, graph: "PortfolioGraph") -> Dict[str, Any]:
        """Build hierarchical node structure."""
        # Hierarchy building logic would go here
        return {}  # Placeholder


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