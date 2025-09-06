"""
Portfolio Graph Implementation
==============================

Main container class for the portfolio hierarchy with graph operations.
"""

from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING, Any
from collections import defaultdict, deque
from datetime import date
import numpy as np
import pandas as pd

from portfolio.metrics import (
    MetricStore, InMemoryMetricStore, 
    WeightedAverage
)
from portfolio.components import PortfolioComponent, PortfolioNode

if TYPE_CHECKING:
    from portfolio.visitors import (
        AggregationVisitor, MultiMetricVisitor,
        ExcessReturnVisitor, FactorRiskDecompositionVisitor
    )
    from portfolio.metrics import Metric, Aggregator, WeightCalculationService
    from risk.estimator import LinearRiskModelEstimator


class PortfolioGraph:
    """Main container for the portfolio hierarchy"""
    
    def __init__(self, root_id: Optional[str] = None, metric_store: Optional[MetricStore] = None):
        self.components: Dict[str, PortfolioComponent] = {}
        self.root_id = root_id
        self._adjacency_list: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
        
        # Metric storage
        self.metric_store = metric_store or InMemoryMetricStore()
    
    @property
    def adjacency_list(self) -> Dict[str, List[str]]:
        """Get adjacency list representation of parent-child relationships."""
        return {parent: list(children) for parent, children in self._adjacency_list.items()}
    
    def add_component(self, component: PortfolioComponent) -> bool:
        """Add a component to the graph"""
        if component.component_id in self.components:
            return False
        
        # Set reference to this graph for visitor pattern support
        component.set_parent_graph(self)
        
        # Set metric store if component doesn't have one
        if component.metric_store is None:
            component.metric_store = self.metric_store
        
        self.components[component.component_id] = component
        return True
    
    def remove_component(self, component_id: str) -> bool:
        """Remove a component and all its connections"""
        if component_id not in self.components:
            return False
        
        component = self.components[component_id]
        
        # Remove from parents
        for parent_id in component.parent_ids:
            if parent_id in self.components:
                parent = self.components[parent_id]
                if isinstance(parent, PortfolioNode):
                    parent.remove_child(component_id)
                self._adjacency_list[parent_id].discard(component_id)
        
        # Remove children connections
        for child_id in component.get_all_children():
            if child_id in self.components:
                self.components[child_id].parent_ids.discard(component_id)
                self._reverse_adjacency[child_id].discard(component_id)
        
        # Clean up adjacency lists
        if component_id in self._adjacency_list:
            del self._adjacency_list[component_id]
        if component_id in self._reverse_adjacency:
            del self._reverse_adjacency[component_id]
        
        # Remove component
        del self.components[component_id]
        
        # Update root if necessary
        if self.root_id == component_id:
            self.root_id = None
        
        return True
    
    def add_edge(self, parent_id: str, child_id: str) -> bool:
        """Add parent-child relationship"""
        if parent_id not in self.components or child_id not in self.components:
            return False
        
        parent = self.components[parent_id]
        child = self.components[child_id]
        
        # Check if parent is a node (not leaf)
        if parent.is_leaf():
            return False
        
        # Check for cycles
        if self._would_create_cycle(parent_id, child_id):
            return False
        
        # Add relationship
        parent.add_child(child_id)
        child.parent_ids.add(parent_id)
        
        # Update adjacency lists
        self._adjacency_list[parent_id].add(child_id)
        self._reverse_adjacency[child_id].add(parent_id)
        
        return True
    
    def remove_edge(self, parent_id: str, child_id: str) -> bool:
        """Remove parent-child relationship"""
        if parent_id not in self.components or child_id not in self.components:
            return False
        
        parent = self.components[parent_id]
        child = self.components[child_id]
        
        parent.remove_child(child_id)
        child.parent_ids.discard(parent_id)
        
        self._adjacency_list[parent_id].discard(child_id)
        self._reverse_adjacency[child_id].discard(parent_id)
        
        return True
    
    def _would_create_cycle(self, parent_id: str, child_id: str) -> bool:
        """Check if adding edge would create a cycle"""
        # Use DFS to check if child_id can reach parent_id
        visited = set()
        stack = [child_id]
        
        while stack:
            current = stack.pop()
            if current == parent_id:
                return True
            
            if current in visited:
                continue
            
            visited.add(current)
            stack.extend(self._adjacency_list[current])
        
        return False
    
    
    def get_path_to_root(self, component_id: str) -> List[str]:
        """Get path from component to root"""
        if component_id not in self.components:
            return []
        
        path = []
        current = component_id
        visited = set()
        
        while current and current not in visited:
            path.append(current)
            visited.add(current)
            
            # Get first parent (assuming tree structure for path)
            parents = self.components[current].parent_ids
            current = next(iter(parents)) if parents else None
        
        return path
    
    def get_all_leaves(self) -> List[str]:
        """Get all leaf component IDs"""
        return [cid for cid, component in self.components.items() 
                if component.is_leaf()]
    
    def get_all_nodes(self) -> List[str]:
        """Get all node component IDs"""
        return [cid for cid, component in self.components.items() 
                if not component.is_leaf()]
    
    def validate_structure(self) -> Tuple[bool, List[str]]:
        """Validate graph structure and return any issues"""
        issues = []
        
        # Check for orphaned components (except root)
        for cid, component in self.components.items():
            if cid != self.root_id and len(component.parent_ids) == 0:
                issues.append(f"Orphaned component: {cid}")
        
        # Check for cycles
        if self._has_cycles():
            issues.append("Graph contains cycles")
        
        # Check weight consistency using visitor pattern
        for cid, component in self.components.items():
            if not component.is_leaf() and len(component.get_all_children()) > 0:
                # Get parent weight from metric store
                parent_weight_metric = self.metric_store.get_metric(cid, 'weight')
                parent_weight = parent_weight_metric.value() if parent_weight_metric else 0.0
                
                # Sum child weights
                child_weights = 0.0
                for child_id in component.get_all_children():
                    if child_id in self.components:
                        child_weight_metric = self.metric_store.get_metric(child_id, 'weight')
                        if child_weight_metric:
                            child_weights += child_weight_metric.value()
                
                # Check weight consistency
                if parent_weight > 0 and abs(child_weights - parent_weight) > 1e-3:
                    issues.append(f"Child weights ({child_weights:.6f}) don't match parent weight ({parent_weight:.6f}) for node {cid}")
        
        return len(issues) == 0, issues
    
    def _has_cycles(self) -> bool:
        """Check if graph has cycles using DFS"""
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for child in self._adjacency_list[node]:
                if dfs(child):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.components:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def prune_empty_nodes(self):
        """Remove nodes with no children"""
        to_remove = []
        for cid, component in self.components.items():
            if not component.is_leaf() and len(component.get_all_children()) == 0:
                to_remove.append(cid)
        
        for cid in to_remove:
            self.remove_component(cid)
    
    def get_subgraph(self, root_component_id: str) -> 'PortfolioGraph':
        """Extract subgraph starting from given component"""
        if root_component_id not in self.components:
            return PortfolioGraph()
        
        subgraph = PortfolioGraph(root_component_id)
        
        # BFS to get all descendants
        queue = deque([root_component_id])
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            
            # Add component to subgraph
            component = self.components[current]
            subgraph.add_component(component)
            
            # Add children to queue
            for child_id in component.get_all_children():
                if child_id in self.components:
                    queue.append(child_id)
        
        # Rebuild edges
        for cid in visited:
            component = self.components[cid]
            if not component.is_leaf():
                for child_id in component.get_all_children():
                    if child_id in visited:
                        subgraph.add_edge(cid, child_id)
        
        return subgraph
    
    # New visitor-based aggregation methods
    def aggregate_metric(self, 
                        root_component_id: str, 
                        metric_name: str, 
                        aggregator=None,
                        weight_metric_name: str = 'weight') -> Optional['Metric']:
        """Aggregate a single metric across hierarchy using visitor pattern"""
        if root_component_id not in self.components:
            return None
        
        if aggregator is None:
            aggregator = WeightedAverage()
        
        from .visitors import AggregationVisitor
        visitor = AggregationVisitor(metric_name, aggregator, self.metric_store, weight_metric_name)
        root_component = self.components[root_component_id]
        
        return visitor.run_on(root_component)
    
    def aggregate_multiple_metrics(self, 
                                  root_component_id: str, 
                                  metric_names: Set[str], 
                                  per_metric_aggregators: Optional[Dict[str, 'Aggregator']] = None,
                                  weight_metric_name: str = 'weight') -> Optional[Dict[str, 'Metric']]:
        """Aggregate multiple metrics simultaneously using visitor pattern"""
        if root_component_id not in self.components:
            return None
        
        if per_metric_aggregators is None:
            per_metric_aggregators = {name: WeightedAverage() for name in metric_names}
        
        from .metrics import MultiMetricAggregator
        from .visitors import MultiMetricVisitor
        multi_aggregator = MultiMetricAggregator(per_metric_aggregators)
        visitor = MultiMetricVisitor(metric_names, multi_aggregator, self.metric_store, weight_metric_name)
        root_component = self.components[root_component_id]
        
        return visitor.run_on(root_component)
    
    def portfolio_returns(self, 
                          root_component_id: str,
                          metric_name: str = 'portfolio_return',
                          weight_metric_name: str = 'portfolio_weight',
                          context: str = 'operational') -> Optional['Metric']:
        """Calculate portfolio returns using visitor pattern with overlay support"""
        if root_component_id not in self.components:
            return None
        from .visitors import AggregationVisitor
        from .metrics import WeightedAverage
        visitor = AggregationVisitor(metric_name, WeightedAverage(), self.metric_store, weight_metric_name, context)
        root_component = self.components[root_component_id]
        return visitor.run_on(root_component)
    
    def benchmark_returns(self, 
                          root_component_id: str,
                          metric_name: str = 'benchmark_return',
                          weight_metric_name: str = 'benchmark_weight',
                          context: str = 'operational') -> Optional['Metric']:
        """Calculate benchmark returns using visitor pattern with overlay support"""
        if root_component_id not in self.components:
            return None
        from .visitors import AggregationVisitor
        from .metrics import WeightedAverage
        visitor = AggregationVisitor(metric_name, WeightedAverage(), self.metric_store, weight_metric_name, context)
        root_component = self.components[root_component_id]
        return visitor.run_on(root_component)
    
    def excess_returns(self, 
                       root_component_id: str,
                       portfolio_metric: str = 'portfolio_return',
                       benchmark_metric: str = 'benchmark_return',
                       portfolio_weight_metric: str = 'portfolio_weight',
                       benchmark_weight_metric: str = 'benchmark_weight') -> Optional['Metric']:
        """Calculate excess returns (Portfolio - Benchmark) using visitor pattern"""
        
        if root_component_id not in self.components:
            return None
        
        portfolio_return = self.portfolio_returns(root_component_id=root_component_id,
                                        metric_name=portfolio_metric,
                                        weight_metric_name=portfolio_weight_metric)
        
        benchmark_return = self.benchmark_returns(root_component_id=root_component_id,
                                        metric_name=benchmark_metric,
                                        weight_metric_name=benchmark_weight_metric)
        

        if portfolio_return is None or benchmark_return is None:
            return None
        
        # Calculate excess return using values and return appropriate metric type
        portfolio_value = portfolio_return.value()
        benchmark_value = benchmark_return.value()
        
        # Calculate excess
        excess_value = portfolio_value - benchmark_value
        
        # Return appropriate metric type
        from .metrics import ScalarMetric, SeriesMetric
        if isinstance(portfolio_return, ScalarMetric):
            return ScalarMetric(excess_value)
        elif isinstance(portfolio_return, SeriesMetric):
            return SeriesMetric(excess_value)
        else:
            return ScalarMetric(0.0)
    
    def calculate_excess_returns(self, 
                               root_component_id: str, 
                               portfolio_metric: str = "portfolio_return", 
                               benchmark_metric: str = "benchmark_return",
                               portfolio_weight_metric: str = 'portfolio_weight',
                               benchmark_weight_metric: str = 'benchmark_weight') -> Optional['Metric']:
        """Calculate excess returns (Portfolio - Benchmark) using visitor pattern with separate weights"""
        if root_component_id not in self.components:
            return None
        
        from .visitors import ExcessReturnVisitor
        visitor = ExcessReturnVisitor(
            portfolio_metric, benchmark_metric, self.metric_store, 
            portfolio_weight_metric, benchmark_weight_metric
        )
        root_component = self.components[root_component_id]
        
        return visitor.run_on(root_component)
    
    
    def create_weight_service(self) -> 'WeightCalculationService':
        """
        Create a WeightCalculationService for this graph.
        
        The service provides cached, pre-populated WeightPathAggregator instances
        that are automatically synchronized with the current graph state.
        
        Returns
        -------
        WeightCalculationService
            Service instance for creating weight path aggregators
        """
        from .metrics import WeightCalculationService
        return WeightCalculationService(self)
    
    def portfolio_weights(self, 
                         component_id: str, 
                         weight_type: str = 'operational',
                         relative_to: str = 'parent',
                         normalize: bool = False) -> Dict[str, float]:
        """
        Get portfolio weights for all descendant leaves of a component.
        
        Parameters
        ----------
        component_id : str
            ID of the component whose descendant leaves to analyze
        weight_type : str, default 'operational'
            The weight context to use:
            - 'operational': For risk/performance calculations (overlays use parent operational weight)
            - 'allocation': For capital allocation (overlays have 0.0 weight)
            - 'reporting', 'accounting': Future extensibility
        relative_to : str, default 'parent'
            How to express weights:
            - 'parent': Weights as stored, context-aware for overlays
            - 'root': Weights relative to the tree root (cumulative through hierarchy)
            - 'specified': Weights exactly as stored in metric store (ignores weight_type)
        normalize : bool, default False
            Whether to normalize weights:
            - False: Return weights as calculated by relative_to
            - True: Normalize weights relative to sum of allocation weights
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping leaf component IDs to their weights
            
        Raises
        ------
        ValueError
            If component_id not found or relative_to parameter invalid
        """
        if component_id not in self.components:
            raise ValueError(f"Component '{component_id}' not found in portfolio graph")
        
        if relative_to not in ['parent', 'root', 'specified']:
            raise ValueError(f"relative_to must be 'parent', 'root', or 'specified', got '{relative_to}'")
        
        # Get all descendant leaves
        descendant_leaves = self._get_descendant_leaves(component_id)
        
        if not descendant_leaves:
            return {}
        
        # Get weights based on relative_to
        if relative_to == 'specified':
            # Get weights exactly as stored (ignore weight_type context)
            weights = self._get_direct_weights_as_stored(descendant_leaves, 'portfolio_weight')
        elif relative_to == 'parent':
            # Direct weight extraction using visitor pattern logic
            weights = self._get_direct_weights(descendant_leaves, 'portfolio_weight', weight_type)
        else:  # relative_to == 'root'
            # Calculate weights relative to the tree root
            if self.root_id is None:
                raise ValueError("Cannot calculate weights relative to root: no root_id set for this graph")
            weights = self._get_weights_relative_to(self.root_id, component_id, 'portfolio_weight', weight_type)
        
        # Apply normalization if requested
        if normalize:
            weights = self._normalize_weights(weights, 'portfolio_weight', weight_type)
        
        return weights

    def benchmark_weights(self, 
                         component_id: str, 
                         weight_type: str = 'operational',
                         relative_to: str = 'parent',
                         normalize: bool = False) -> Dict[str, float]:
        """
        Get benchmark weights for all descendant leaves of a component.
        
        Parameters
        ----------
        component_id : str
            ID of the component whose descendant leaves to analyze
        weight_type : str, default 'operational'
            The weight context to use:
            - 'operational': For risk/performance calculations (overlays use parent operational weight)
            - 'allocation': For capital allocation (overlays have 0.0 weight)
            - 'reporting', 'accounting': Future extensibility
        relative_to : str, default 'parent'
            How to express weights:
            - 'parent': Weights as stored, context-aware for overlays
            - 'root': Weights relative to the tree root (cumulative through hierarchy)
            - 'specified': Weights exactly as stored in metric store (ignores weight_type)
        normalize : bool, default False
            Whether to normalize weights:
            - False: Return weights as calculated by relative_to
            - True: Normalize weights relative to sum of allocation weights
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping leaf component IDs to their weights
            
        Raises
        ------
        ValueError
            If component_id not found or relative_to parameter invalid
        """
        if component_id not in self.components:
            raise ValueError(f"Component '{component_id}' not found in portfolio graph")
        
        if relative_to not in ['parent', 'root', 'specified']:
            raise ValueError(f"relative_to must be 'parent', 'root', or 'specified', got '{relative_to}'")
        
        # Get all descendant leaves
        descendant_leaves = self._get_descendant_leaves(component_id)
        
        if not descendant_leaves:
            return {}
        
        # Get weights based on relative_to
        if relative_to == 'specified':
            # Get weights exactly as stored (ignore weight_type context)
            weights = self._get_direct_weights_as_stored(descendant_leaves, 'benchmark_weight')
        elif relative_to == 'parent':
            # Direct weight extraction using visitor pattern logic
            weights = self._get_direct_weights(descendant_leaves, 'benchmark_weight', weight_type)
        else:  # relative_to == 'root'
            # Calculate weights relative to the tree root
            if self.root_id is None:
                raise ValueError("Cannot calculate weights relative to root: no root_id set for this graph")
            weights = self._get_weights_relative_to(self.root_id, component_id, 'benchmark_weight', weight_type)
        
        # Apply normalization if requested
        if normalize:
            weights = self._normalize_weights(weights, 'benchmark_weight', weight_type)
        
        return weights

    def collect_metrics(self, 
                       component_id: str, 
                       metric_name: str,
                       include_nodes: bool = False,
                       when: Optional[date] = None) -> Dict[str, Any]:
        """
        Collect a specific metric from all descendant components.
        
        This method retrieves the specified metric from all descendants of the given
        component and returns them as a dictionary mapping component IDs to their metric values.
        
        Parameters
        ----------
        component_id : str
            ID of the component whose descendants to collect metrics from
        metric_name : str
            Name of the metric to collect (e.g., 'portfolio_return', 'portfolio_weight')
        include_nodes : bool, default False
            Whether to include intermediate nodes or only leaves
            - False: Only collect from leaf components
            - True: Collect from both intermediate nodes and leaves
        when : date, optional
            Optional date parameter to pass to metric.value() for time-series metrics
            
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping component IDs to their metric values.
            Only includes components that have the specified metric.
            Value types depend on the metric type:
            - ScalarMetric: float
            - SeriesMetric: pandas.Series
            - ArrayMetric: numpy.ndarray
            - ObjectMetric: Any
            
        Raises
        ------
        ValueError
            If component_id not found in the graph
            
        Examples
        --------
        # Collect portfolio returns from all leaves under equity
        >>> returns = graph.collect_metrics('equity', 'portfolio_return')
        {'tech': 0.05, 'healthcare': 0.03, 'energy': 0.02}
        
        # Collect weights including intermediate nodes
        >>> weights = graph.collect_metrics('portfolio', 'portfolio_weight', include_nodes=True)
        {'equity': 0.8, 'bonds': 0.2, 'tech': 0.48, 'healthcare': 0.32}
        
        # Collect time series metrics as of specific date
        >>> returns_ts = graph.collect_metrics('equity', 'daily_returns', when=date(2024, 1, 15))
        """
        if component_id not in self.components:
            raise ValueError(f"Component '{component_id}' not found in portfolio graph")
        
        # Get appropriate descendants based on include_nodes parameter
        if include_nodes:
            descendant_ids = self._get_all_descendants(component_id)
        else:
            descendant_ids = self._get_descendant_leaves(component_id)
        
        # Collect metrics from descendants
        metrics_dict = {}
        
        for desc_id in descendant_ids:
            # Check if the metric exists for this component
            metric = self.metric_store.get_metric(desc_id, metric_name)
            if metric is not None:
                # Get the metric value, optionally filtered by date
                try:
                    value = metric.value(when=when)
                    metrics_dict[desc_id] = value
                except Exception as e:
                    # Log warning but continue with other components
                    # This handles cases where metric.value() might fail
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to get metric '{metric_name}' for component '{desc_id}': {e}")
        
        return metrics_dict

    def _get_direct_weights(self, leaf_ids: List[str], weight_metric: str, weight_type: str) -> Dict[str, float]:
        """
        Get direct weights for leaf components using visitor pattern logic.
        
        Parameters
        ----------
        leaf_ids : list of str
            List of leaf component IDs
        weight_metric : str
            Base weight metric name ('portfolio_weight' or 'benchmark_weight')
        weight_type : str
            Weight context ('operational', 'allocation', etc.)
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping leaf IDs to their weights
        """
        weights = {}
        
        for leaf_id in leaf_ids:
            component = self.components[leaf_id]
            
            # Use visitor pattern logic for context-aware weight extraction
            if hasattr(component, 'is_overlay') and getattr(component, 'is_overlay', False):
                if weight_type == 'operational':
                    # For operational context, use parent operational weight
                    weight = component.get_operational_weight(weight_metric)
                else:
                    # For allocation context, overlay components have 0.0 weight
                    weight = 0.0
            else:
                # Standard component logic
                if component.metric_store:
                    weight_metric_obj = component.metric_store.get_metric(component.component_id, weight_metric)
                    weight = weight_metric_obj.value() if weight_metric_obj else 0.0
                else:
                    weight = 0.0
            
            weights[leaf_id] = weight
        
        return weights

    def _get_direct_weights_as_stored(self, leaf_ids: List[str], weight_metric: str) -> Dict[str, float]:
        """
        Get weights for leaf components exactly as stored in metric store.
        
        This method ignores weight_type context and returns raw stored values,
        useful for the 'specified' relative_to mode.
        
        Parameters
        ----------
        leaf_ids : list of str
            List of leaf component IDs
        weight_metric : str
            Base weight metric name ('portfolio_weight' or 'benchmark_weight')
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping leaf IDs to their stored weights
        """
        weights = {}
        
        for leaf_id in leaf_ids:
            component = self.components[leaf_id]
            
            # Get weight exactly as stored, ignoring overlay logic
            if component.metric_store:
                weight_metric_obj = component.metric_store.get_metric(component.component_id, weight_metric)
                weight = weight_metric_obj.value() if weight_metric_obj else 0.0
            else:
                weight = 0.0
            
            weights[leaf_id] = weight
        
        return weights

    def _normalize_weights(self, weights: Dict[str, float], weight_metric: str, weight_type: str) -> Dict[str, float]:
        """
        Normalize weights relative to sum of allocation weights.
        
        Parameters
        ----------
        weights : Dict[str, float]
            Raw weights to normalize
        weight_metric : str
            Base weight metric name ('portfolio_weight' or 'benchmark_weight')
        weight_type : str
            Weight context ('operational', 'allocation', etc.)
            
        Returns
        -------
        Dict[str, float]
            Normalized weights
        """
        if not weights:
            return weights
        
        # Get allocation weights for normalization base
        allocation_weights = {}
        for comp_id in weights.keys():
            component = self.components[comp_id]
            if component.metric_store:
                alloc_metric = component.metric_store.get_metric(comp_id, weight_metric)
                allocation_weights[comp_id] = alloc_metric.value() if alloc_metric else 0.0
            else:
                allocation_weights[comp_id] = 0.0
        
        allocation_sum = sum(allocation_weights.values())
        if allocation_sum == 0:
            return weights  # Can't normalize, return as-is
        
        # Normalize based on allocation sum
        normalized = {comp_id: allocation_weights[comp_id] / allocation_sum for comp_id in weights.keys()}
        
        # For operational context, set overlays to 1.0
        if weight_type == 'operational':
            for comp_id in weights.keys():
                component = self.components[comp_id]
                if hasattr(component, 'is_overlay') and getattr(component, 'is_overlay', False):
                    normalized[comp_id] = 1.0
        
        return normalized

    def _get_weights_relative_to(self, target_root_id: str, component_id: str, 
                               weight_metric: str, weight_type: str) -> Dict[str, float]:
        """
        Get weights of descendant leaves relative to a specified target root component.
        
        This method calculates the effective weights of all descendant leaves of component_id
        as they would appear from the perspective of target_root_id. This enables flexible
        weight calculations relative to any point in the hierarchy.
        
        Parameters
        ----------
        target_root_id : str
            ID of the component to use as the reference root for weight calculations
        component_id : str
            ID of the component whose descendants to analyze
        weight_metric : str
            Base weight metric name ('portfolio_weight' or 'benchmark_weight')
        weight_type : str
            Weight context ('operational', 'allocation', etc.)
            
        Returns
        -------
        Dict[str, float]
            Dictionary mapping leaf IDs to their weights relative to target_root_id
            
        Raises
        ------
        ValueError
            If target_root_id or component_id not found, or if target_root_id cannot reach component_id
        """
        # Validate inputs
        if target_root_id not in self.components:
            raise ValueError(f"Target root component '{target_root_id}' not found in portfolio graph")
        if component_id not in self.components:
            raise ValueError(f"Component '{component_id}' not found in portfolio graph")
        
        # Get descendant leaves of the component
        leaf_ids = self._get_descendant_leaves(component_id)
        if not leaf_ids:
            return {}
        
        weights = {}
        
        for leaf_id in leaf_ids:
            # Get path from target root to leaf
            path = self._get_path_from_to(target_root_id, leaf_id)
            if not path:
                # If no path exists from target_root to this leaf, weight is 0
                weights[leaf_id] = 0.0
                continue
            
            # Calculate cumulative weight along path
            cumulative_weight = 1.0
            
            for path_component_id in path:
                component = self.components[path_component_id]
                
                # Apply visitor pattern logic at each level
                if hasattr(component, 'is_overlay') and getattr(component, 'is_overlay', False):
                    if weight_type == 'operational':
                        # For path aggregation, overlay components don't reduce weight (they inherit parent's allocation)
                        # The parent's weight is already counted in the path, so overlay contributes 1.0
                        component_weight = 1.0
                    else:
                        # For allocation context, overlays don't get capital allocation
                        component_weight = 0.0
                else:
                    if component.metric_store:
                        weight_metric_obj = component.metric_store.get_metric(path_component_id, weight_metric)
                        component_weight = weight_metric_obj.value() if weight_metric_obj else 1.0
                    else:
                        component_weight = 1.0
                
                cumulative_weight *= component_weight
            
            weights[leaf_id] = cumulative_weight
        
        return weights

    def _get_path_from_to(self, start_id: str, end_id: str) -> Optional[List[str]]:
        """
        Get path from start component to end component.
        
        Parameters
        ----------
        start_id : str
            Starting component ID
        end_id : str
            Ending component ID
            
        Returns
        -------
        Optional[List[str]]
            List of component IDs forming the path, or None if no path exists
        """
        if start_id == end_id:
            return [start_id]
        
        # Use BFS to find path
        from collections import deque
        queue = deque([(start_id, [start_id])])
        visited = set()
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id in visited:
                continue
            visited.add(current_id)
            
            if current_id == end_id:
                return path
            
            # Add children to queue
            current_component = self.components.get(current_id)
            if current_component and not current_component.is_leaf():
                for child_id in current_component.get_all_children():
                    if child_id in self.components and child_id not in visited:
                        queue.append((child_id, path + [child_id]))
        
        return None
    
    def _get_descendant_leaves(self, node_id: str) -> List[str]:
        """Get all leaf descendants of a node"""
        from collections import deque
        
        if node_id not in self.components:
            return []
        
        component = self.components[node_id]
        if component.is_leaf():
            return [node_id]
        
        descendants = []
        queue = deque([node_id])
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            current_component = self.components[current]
            if current_component.is_leaf():
                descendants.append(current)
            else:
                queue.extend(current_component.get_all_children())
        
        return descendants
    
    def _get_all_descendants(self, node_id: str) -> List[str]:
        """Get all descendants of a node (both nodes and leaves)"""
        from collections import deque
        
        if node_id not in self.components:
            return []
        
        component = self.components[node_id]
        if component.is_leaf():
            return [node_id]
        
        descendants = []
        queue = deque([node_id])
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            current_component = self.components[current]
            # Include both leaves and intermediate nodes (but skip the starting node)
            if current != node_id:
                descendants.append(current)
            
            # Add children to queue if it's not a leaf
            if not current_component.is_leaf():
                queue.extend(current_component.get_all_children())
        
        return descendants
    
    def print_tree(self, 
                   root_component_id: Optional[str] = None,
                   portfolio_weight_metric: str = 'portfolio_weight',
                   benchmark_weight_metric: str = 'benchmark_weight',
                   show_metrics: Optional[List[str]] = None,
                   show_active_weights: bool = True,
                   max_depth: Optional[int] = None,
                   return_string: bool = False) -> Optional[str]:
        """
        Create a text-based visualization of the portfolio tree structure
        
        Args:
            root_component_id: Component to start from (defaults to self.root_id)
            portfolio_weight_metric: Name of the portfolio weight metric (default: 'weight')
            benchmark_weight_metric: Name of the benchmark weight metric (default: 'benchmark_weight')
            show_metrics: List of additional metric names to display
            show_active_weights: Whether to show active weights (Portfolio - Benchmark)
            max_depth: Maximum depth to traverse (None for unlimited)
            return_string: If True, return the tree as string instead of printing
            
        Returns:
            Optional[str]: Tree representation if return_string=True, None otherwise
        """
        if root_component_id is None:
            root_component_id = self.root_id
            
        if root_component_id is None or root_component_id not in self.components:
            error_msg = f"Invalid root component: {root_component_id}"
            if return_string:
                return error_msg
            print(error_msg)
            return None
        
        if show_metrics is None:
            show_metrics = []
        
        tree_lines = []
        
        def _build_tree_recursive(component_id: str, prefix: str = "", is_last: bool = True, depth: int = 0):
            """Recursively build tree representation"""
            if max_depth is not None and depth > max_depth:
                return
                
            component = self.components[component_id]
            
            # Get weights from metric store
            portfolio_weight = 0.0
            benchmark_weight = 0.0
            
            if self.metric_store:
                port_metric = self.metric_store.get_metric(component_id, portfolio_weight_metric)
                bench_metric = self.metric_store.get_metric(component_id, benchmark_weight_metric)
                
                if port_metric:
                    portfolio_weight = port_metric.value()
                if bench_metric:
                    benchmark_weight = bench_metric.value()
            
            # Calculate active weight
            active_weight = portfolio_weight - benchmark_weight
            
            # Build weight display
            weight_parts = [f"P: {portfolio_weight:.4f}", f"B: {benchmark_weight:.4f}"]
            if show_active_weights:
                sign = "+" if active_weight >= 0 else ""
                weight_parts.append(f"A: {sign}{active_weight:.4f}")
            
            weight_str = f"[{' | '.join(weight_parts)}]"
            
            # Build additional metrics display
            metrics_str = ""
            if show_metrics:
                metric_values = []
                for metric_name in show_metrics:
                    if self.metric_store:
                        metric = self.metric_store.get_metric(component_id, metric_name)
                        if metric:
                            value = metric.value()
                            if isinstance(value, float):
                                metric_values.append(f"{metric_name}: {value:.4f}")
                            else:
                                metric_values.append(f"{metric_name}: {value}")
                        else:
                            metric_values.append(f"{metric_name}: N/A")
                
                if metric_values:
                    metrics_str = f" {{{', '.join(metric_values)}}}"
            
            # Component type indicator
            type_indicator = "🔸" if component.is_leaf() else "📁"
            
            # Build the line for this component
            connector = "└── " if is_last else "├── "
            line = f"{prefix}{connector}{type_indicator} {component.name} ({component_id}) {weight_str}{metrics_str}"
            tree_lines.append(line)
            
            # Recursively add children
            if not component.is_leaf():
                children = list(component.get_all_children())
                for i, child_id in enumerate(children):
                    if child_id in self.components:
                        is_last_child = (i == len(children) - 1)
                        child_prefix = prefix + ("    " if is_last else "│   ")
                        _build_tree_recursive(child_id, child_prefix, is_last_child, depth + 1)
        
        # Start building the tree
        root_component = self.components[root_component_id]
        tree_lines.append(f"Portfolio Tree Structure (Root: {root_component.name})")
        tree_lines.append("=" * 60)
        tree_lines.append("")
        
        _build_tree_recursive(root_component_id)
        
        # Add legend
        tree_lines.append("")
        tree_lines.append("Legend:")
        tree_lines.append("  📁 = Portfolio Node (container)")
        tree_lines.append("  🔸 = Portfolio Leaf (security)")
        tree_lines.append(f"  P = Portfolio Weight ({portfolio_weight_metric})")
        tree_lines.append(f"  B = Benchmark Weight ({benchmark_weight_metric})")
        if show_active_weights:
            tree_lines.append("  A = Active Weight (P - B)")
        
        result = "\n".join(tree_lines)
        
        if return_string:
            return result
        else:
            print(result)
            return None
    
    
# Decision attribution methods are available through the decision_attribution module
# Usage: from portfolio.decision_attribution import HierarchicalDecisionAnalyzer, PortfolioDecision

# Risk analysis methods are available through the risk_analyzer module
# Usage: from portfolio.risk_analyzer import PortfolioRiskAnalyzer


# Add convenience methods for enhanced PortfolioBuilder compatibility
def get_component_weight(self, component_id: str, weight_type: str) -> Optional[float]:
    """Get weight for a component from metric store."""
    weight_metric = self.metric_store.get_metric(component_id, weight_type)
    return weight_metric.value() if weight_metric else None

def get_active_weight(self, component_id: str) -> Optional[float]:
    """Get active weight (portfolio - benchmark) for a component."""
    port_weight = self.get_component_weight(component_id, 'portfolio_weight')
    bench_weight = self.get_component_weight(component_id, 'benchmark_weight')
    
    if port_weight is not None and bench_weight is not None:
        return port_weight - bench_weight
    return None

# Bind methods to PortfolioGraph class
PortfolioGraph.get_component_weight = get_component_weight
PortfolioGraph.get_active_weight = get_active_weight

# Import from builders module for backward compatibility and enhanced construction methods
from .builders import PortfolioBuilder, create_portfolio_graph_class_methods

# Apply enhanced construction methods to PortfolioGraph class
_enhanced_methods = create_portfolio_graph_class_methods()
PortfolioGraph.from_hierarchy_df = _enhanced_methods['from_hierarchy_df']
PortfolioGraph.from_dict = _enhanced_methods['from_dict']
PortfolioGraph.select_components = _enhanced_methods['select_components']
PortfolioGraph.update_subtree_weights = _enhanced_methods['update_subtree_weights']