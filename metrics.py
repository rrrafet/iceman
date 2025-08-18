"""
Portfolio Metrics and Component Types
=====================================

Core data structures for portfolio hierarchy components, including metric abstractions
and aggregation strategies following the visitor pattern design.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Iterable, Tuple, List, TYPE_CHECKING
from datetime import date
import pandas as pd
import numpy as np
from ..core.transformable import Transformable

if TYPE_CHECKING:
    from .graph import PortfolioGraph
    from .components import PortfolioComponent


class ComponentMetrics(Transformable):
    """Container for portfolio/benchmark/active metrics"""
    
    def __init__(self, 
                 data: Optional[pd.DataFrame] = None,
                 forward_returns: Optional[float] = None,
                 forward_risk: Optional[float] = None,
                 scaling_factor: float = 1.0,
                 weight: float = 0.0,
                 forward_specific_returns: Optional[float] = None,
                 forward_specific_risk: Optional[float] = None,
                 metadata: Optional[Dict] = None):
        """
        Initialize ComponentMetrics with optional DataFrame and metric fields.
        
        Args:
            data: Optional pandas DataFrame for transformable operations
            forward_returns: Expected forward returns
            forward_risk: Expected forward risk
            scaling_factor: Scaling factor for metrics
            weight: Portfolio weight
            forward_specific_returns: Expected specific returns
            forward_specific_risk: Expected specific risk
            metadata: Optional dictionary for additional metadata
        """
        # Initialize Transformable with data or empty DataFrame
        if data is not None:
            super().__init__(data)
        else:
            super().__init__(pd.DataFrame())
        
        # Set metric fields
        self.forward_returns = forward_returns
        self.forward_risk = forward_risk
        self.scaling_factor = scaling_factor
        self.weight = weight
        self.forward_specific_returns = forward_specific_returns
        self.forward_specific_risk = forward_specific_risk
        self.metadata = metadata if metadata is not None else {}
    
    def transform(self) -> pd.DataFrame:
        """
        Transform method required by Transformable interface.
        
        Returns:
            Copy of current data
        """
        return self.data.copy()
    
    def to_metrics_dict(self) -> Dict[str, 'Metric']:
        """Convert ComponentMetrics to dictionary of Metric objects"""
        metrics = {}
        
        if self.forward_returns is not None:
            metrics["forward_returns"] = ScalarMetric(self.forward_returns)
        if self.forward_risk is not None:
            metrics["forward_risk"] = ScalarMetric(self.forward_risk)
        if self.weight is not None:
            metrics["weight"] = ScalarMetric(self.weight)
        if self.scaling_factor is not None:
            metrics["scaling_factor"] = ScalarMetric(self.scaling_factor)
        if self.forward_specific_returns is not None:
            metrics["forward_specific_returns"] = ScalarMetric(self.forward_specific_returns)
        if self.forward_specific_risk is not None:
            metrics["forward_specific_risk"] = ScalarMetric(self.forward_specific_risk)
        
        return metrics



class Metric(ABC):
    """Abstract base class for all portfolio metrics"""
    
    @abstractmethod
    def value(self, when: Optional[date] = None) -> Any:
        """Get metric value, optionally at a specific date"""
        pass
    
    @abstractmethod
    def copy(self) -> 'Metric':
        """Create a copy of this metric"""
        pass


class ScalarMetric(Metric):
    """Metric containing a single scalar value"""
    
    def __init__(self, value: float):
        self._value = float(value)
    
    def value(self, when: Optional[date] = None) -> float:
        """Return scalar value (when parameter ignored)"""
        return self._value
    
    def copy(self) -> 'ScalarMetric':
        return ScalarMetric(self._value)
    
    def __repr__(self):
        return f"ScalarMetric({self._value})"


class SeriesMetric(Metric):
    """Metric containing a time series of values"""
    
    def __init__(self, series: pd.Series):
        if not isinstance(series, pd.Series):
            raise ValueError("SeriesMetric requires a pandas Series")
        self._series = series.copy()
    
    def value(self, when: Optional[date] = None) -> pd.Series:
        """Return series, optionally filtered by date"""
        if when is None:
            return self._series.copy()
        
        # Filter series up to the specified date
        if isinstance(self._series.index, pd.DatetimeIndex):
            mask = self._series.index.date <= when
            return self._series[mask].copy()
        else:
            # If index is not datetime, return full series
            return self._series.copy()
    
    def copy(self) -> 'SeriesMetric':
        return SeriesMetric(self._series)
    
    def __repr__(self):
        return f"SeriesMetric(length={len(self._series)})"


class ArrayMetric(Metric):
    """Metric containing a numpy array of values"""
    
    def __init__(self, array: np.ndarray):
        if not isinstance(array, np.ndarray):
            raise ValueError("ArrayMetric requires a numpy array")
        self._array = array.copy()
    
    def value(self, when: Optional[date] = None) -> np.ndarray:
        """Return array (when parameter ignored for arrays)"""
        return self._array.copy()
    
    def copy(self) -> 'ArrayMetric':
        return ArrayMetric(self._array)
    
    def __repr__(self):
        return f"ArrayMetric(shape={self._array.shape})"


class ObjectMetric(Metric):
    """Metric containing an arbitrary Python object"""
    
    def __init__(self, obj: Any):
        self._obj = obj
    
    def value(self, when: Optional[date] = None) -> Any:
        """Return stored object (when parameter ignored for objects)"""
        return self._obj
    
    def copy(self) -> 'ObjectMetric':
        # Note: This performs a shallow copy of the object
        # For deep copying, consider using copy.deepcopy if needed
        return ObjectMetric(self._obj)
    
    def __repr__(self):
        return f"ObjectMetric(type={type(self._obj).__name__})"


class MetricStore(ABC):
    """Abstract interface for metric storage backend"""
    
    @abstractmethod
    def get_metric(self, component_id: str, metric_name: str) -> Optional[Metric]:
        """Retrieve a metric for a component"""
        pass
    
    @abstractmethod
    def set_metric(self, component_id: str, metric_name: str, metric: Metric) -> None:
        """Store a metric for a component"""
        pass
    
    @abstractmethod
    def has_metric(self, component_id: str, metric_name: str) -> bool:
        """Check if metric exists for component"""
        pass
    
    @abstractmethod
    def get_all_metrics(self, component_id: str) -> Dict[str, Metric]:
        """Get all metrics for a component"""
        pass


class InMemoryMetricStore(MetricStore):
    """Simple in-memory metric store implementation"""
    
    def __init__(self):
        self._store: Dict[str, Dict[str, Metric]] = {}
    
    def get_metric(self, component_id: str, metric_name: str) -> Optional[Metric]:
        return self._store.get(component_id, {}).get(metric_name)
    
    def set_metric(self, component_id: str, metric_name: str, metric: Metric) -> None:
        if component_id not in self._store:
            self._store[component_id] = {}
        self._store[component_id][metric_name] = metric
    
    def has_metric(self, component_id: str, metric_name: str) -> bool:
        return metric_name in self._store.get(component_id, {})
    
    def get_all_metrics(self, component_id: str) -> Dict[str, Metric]:
        return self._store.get(component_id, {}).copy()


class Aggregator(ABC):
    """Abstract base class for metric aggregation strategies"""
    
    @abstractmethod
    def combine(self, inputs: Iterable[Tuple[float, Metric]]) -> Metric:
        """Combine weighted metrics into a single result"""
        pass


class Sum(Aggregator):
    """Sum aggregation strategy"""
    
    def combine(self, inputs: Iterable[Tuple[float, Metric]]) -> Metric:
        weighted_values = []
        first_metric = None
        
        for weight, metric in inputs:
            if first_metric is None:
                first_metric = metric
            
            if isinstance(metric, ScalarMetric):
                weighted_values.append(weight * metric.value())
            elif isinstance(metric, SeriesMetric):
                series = metric.value()
                weighted_values.append(weight * series)
            else:
                raise ValueError(f"Unsupported metric type: {type(metric)}")
        
        if not weighted_values:
            return ScalarMetric(0.0)
        
        # Determine return type based on first metric
        if isinstance(first_metric, ScalarMetric):
            return ScalarMetric(sum(weighted_values))
        elif isinstance(first_metric, SeriesMetric):
            result_series = sum(weighted_values)
            return SeriesMetric(result_series)
        else:
            return ScalarMetric(0.0)


class WeightedAverage(Aggregator):
    """Weighted average aggregation strategy with proper weight normalization"""
    
    def __init__(self, normalize_weights: bool = True):
        """
        Initialize WeightedAverage aggregator.
        
        Parameters
        ----------
        normalize_weights : bool, default True
            Whether to normalize weights to sum to 1.0 before aggregation
        """
        self.normalize_weights = normalize_weights
    
    def combine(self, inputs: Iterable[Tuple[float, Metric]]) -> Metric:
        inputs_list = list(inputs)
        
        if not inputs_list:
            return ScalarMetric(0.0)
        
        weights = [weight for weight, _ in inputs_list]
        metrics = [metric for _, metric in inputs_list]
        
        # Normalize weights if requested
        if self.normalize_weights:
            total_weight = sum(weights)
            if total_weight == 0:
                # If all weights are zero, use equal weighting
                weights = [1.0 / len(weights) for _ in weights]
            else:
                weights = [w / total_weight for w in weights]
        
        weighted_values = []
        first_metric = metrics[0]
        
        for weight, metric in zip(weights, metrics):
            if isinstance(metric, ScalarMetric):
                weighted_values.append(weight * metric.value())
            elif isinstance(metric, SeriesMetric):
                series = metric.value()
                weighted_values.append(weight * series)
            else:
                raise ValueError(f"Unsupported metric type: {type(metric)}")

        # Determine return type based on first metric
        if isinstance(first_metric, ScalarMetric):
            result = sum(weighted_values)
            return ScalarMetric(result)
        elif isinstance(first_metric, SeriesMetric):
            # Handle series aggregation properly
            if len(weighted_values) == 1:
                result_series = weighted_values[0]
            else:
                # Sum weighted series
                result_series = weighted_values[0]
                for weighted_series in weighted_values[1:]:
                    result_series = result_series + weighted_series
            return SeriesMetric(result_series)
        else:
            print("Unsupported metric type for weighted average. Returning zero.")
            return ScalarMetric(0.0)


class ExcessReturnAgg(Aggregator):
    """Excess return aggregation strategy (Portfolio - Benchmark)"""
    
    def __init__(self, portfolio_metric: str = "port_ret", benchmark_metric: str = "bench_ret"):
        self.portfolio_metric = portfolio_metric
        self.benchmark_metric = benchmark_metric
    
    def combine(self, inputs: Iterable[Tuple[float, Metric]]) -> Metric:
        # This is a specialized aggregator that expects inputs to be tuples of
        # (weight, dict_of_metrics) where the dict contains both portfolio and benchmark metrics
        portfolio_inputs = []
        benchmark_inputs = []
        
        for weight, metric_dict in inputs:
            if isinstance(metric_dict, dict):
                if self.portfolio_metric in metric_dict:
                    portfolio_inputs.append((weight, metric_dict[self.portfolio_metric]))
                if self.benchmark_metric in metric_dict:
                    benchmark_inputs.append((weight, metric_dict[self.benchmark_metric]))
        
        # Use weighted average for both portfolio and benchmark
        wa_agg = WeightedAverage()
        portfolio_result = wa_agg.combine(portfolio_inputs)
        benchmark_result = wa_agg.combine(benchmark_inputs)
        
        # Calculate excess return
        if isinstance(portfolio_result, ScalarMetric) and isinstance(benchmark_result, ScalarMetric):
            excess = portfolio_result.value() - benchmark_result.value()
            return ScalarMetric(excess)
        elif isinstance(portfolio_result, SeriesMetric) and isinstance(benchmark_result, SeriesMetric):
            excess_series = portfolio_result.value() - benchmark_result.value()
            return SeriesMetric(excess_series)
        else:
            return ScalarMetric(0.0)




class MultiMetricAggregator(Aggregator):
    """Aggregator that handles multiple metrics with different strategies"""
    
    def __init__(self, per_metric_aggregators: Dict[str, Aggregator]):
        self.per_metric_aggregators = per_metric_aggregators
    
    def combine(self, inputs: Iterable[Tuple[float, Dict[str, Metric]]]) -> Dict[str, Metric]:
        # Group inputs by metric name
        metric_inputs = {}
        
        for weight, metrics_dict in inputs:
            for metric_name, metric in metrics_dict.items():
                if metric_name not in metric_inputs:
                    metric_inputs[metric_name] = []
                metric_inputs[metric_name].append((weight, metric))
        
        # Apply appropriate aggregator to each metric
        results = {}
        for metric_name, metric_data in metric_inputs.items():
            if metric_name in self.per_metric_aggregators:
                aggregator = self.per_metric_aggregators[metric_name]
                results[metric_name] = aggregator.combine(metric_data)
        
        return results


class WeightPathAggregator:
    """
    Aggregator for calculating hierarchical weight path matrices.
    
    This class computes how weights flow from each node to all descendant leaves 
    through the portfolio hierarchy, creating path matrices for proper risk decomposition.
    """
    
    def __init__(self):
        """Initialize the weight path aggregator."""
        # Storage for hierarchical relationships
        self._parent_child_map: Dict[str, list] = {}  # parent_id -> [child_ids]
        self._child_parent_map: Dict[str, str] = {}   # child_id -> parent_id
        self._node_weights: Dict[str, Dict[str, float]] = {}  # node_id -> {type: weight}
        self._leaf_nodes: set = set()
        
        # Computed path matrices
        self._path_matrices: Dict[str, Dict[str, Dict[str, float]]] = {}  # node_id -> {type: {leaf_id: weight}}
        
    def add_node_relationship(self, parent_id: str, child_id: str) -> None:
        """Add a parent-child relationship to the hierarchy."""
        if parent_id not in self._parent_child_map:
            self._parent_child_map[parent_id] = []
        self._parent_child_map[parent_id].append(child_id)
        self._child_parent_map[child_id] = parent_id
    
    def set_node_weight(self, node_id: str, weight_type: str, weight: float) -> None:
        """Set the weight for a node for a specific weight type (portfolio, benchmark)."""
        if node_id not in self._node_weights:
            self._node_weights[node_id] = {}
        self._node_weights[node_id][weight_type] = weight
    
    def mark_as_leaf(self, node_id: str) -> None:
        """Mark a node as a leaf node."""
        self._leaf_nodes.add(node_id)
    
    def calculate_path_matrices(self) -> None:
        """Calculate path matrices for all nodes to their descendant leaves."""
        # Find all nodes that have descendants (intermediate and root nodes)
        nodes_with_children = set(self._parent_child_map.keys())
        
        for node_id in nodes_with_children:
            self._path_matrices[node_id] = {}
            
            # Get all descendant leaves for this node
            descendant_leaves = self._get_descendant_leaves(node_id)
            
            # Calculate path matrices for each weight type
            for weight_type in ['portfolio', 'benchmark']:
                self._path_matrices[node_id][weight_type] = self._calculate_path_weights(
                    node_id, descendant_leaves, weight_type
                )
    
    def _get_descendant_leaves(self, node_id: str) -> list:
        """Get all descendant leaf nodes for a given node."""
        descendants = []
        
        def _traverse(current_node):
            if current_node in self._leaf_nodes:
                descendants.append(current_node)
            elif current_node in self._parent_child_map:
                for child in self._parent_child_map[current_node]:
                    _traverse(child)
        
        _traverse(node_id)
        return descendants
    
    def _calculate_path_weights(self, root_node: str, leaf_nodes: list, weight_type: str) -> Dict[str, float]:
        """Calculate the effective weight contribution from root_node to each leaf."""
        path_weights = {}
        
        for leaf_id in leaf_nodes:
            # Find the path from root_node to leaf_id
            path = self._find_path(root_node, leaf_id)
            if path:
                # Calculate cumulative weight along the path
                cumulative_weight = 1.0
                
                # Multiply weights along the entire path (including root node)
                for node_id in path:
                    node_weight = self._node_weights.get(node_id, {}).get(weight_type, 1.0)
                    cumulative_weight *= node_weight
                
                path_weights[leaf_id] = cumulative_weight
            else:
                path_weights[leaf_id] = 0.0
        
        return path_weights
    
    def _find_path(self, start_node: str, end_node: str) -> Optional[list]:
        """Find path from start_node to end_node in the hierarchy."""
        if start_node == end_node:
            return [start_node]
        
        # Use BFS to find path
        from collections import deque
        queue = deque([(start_node, [start_node])])
        visited = set()
        
        while queue:
            current_node, path = queue.popleft()
            
            if current_node in visited:
                continue
            visited.add(current_node)
            
            if current_node == end_node:
                return path
            
            # Add children to queue
            if current_node in self._parent_child_map:
                for child in self._parent_child_map[current_node]:
                    if child not in visited:
                        queue.append((child, path + [child]))
        
        return None
    
    def get_path_matrix(self, node_id: str, weight_type: str) -> Optional[Dict[str, float]]:
        """Get the path matrix for a node and weight type."""
        return self._path_matrices.get(node_id, {}).get(weight_type)
    
    def get_effective_weights(self, node_id: str, weight_type: str) -> Optional[np.ndarray]:
        """Get effective weights as numpy array for a node's descendants."""
        path_matrix = self.get_path_matrix(node_id, weight_type)
        if path_matrix is None:
            return None
        
        # Get descendant leaves in order
        descendant_leaves = self._get_descendant_leaves(node_id)
        
        # Create weight array in leaf order
        weights = np.array([path_matrix.get(leaf_id, 0.0) for leaf_id in descendant_leaves])
        
        return weights
    
    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1.0."""
        total = np.sum(weights)
        if total > 0:
            return weights / total
        else:
            # Equal weights if all are zero
            return np.ones_like(weights) / len(weights) if len(weights) > 0 else weights
    
    def get_descendant_leaves(self, node_id: str) -> list:
        """Public method to get descendant leaves for a node."""
        return self._get_descendant_leaves(node_id)


class WeightCalculationService:
    """
    Service for creating and caching WeightPathAggregator instances.
    
    This service manages weight path calculations for portfolio hierarchies,
    providing caching and automatic synchronization with the PortfolioGraph state.
    """
    
    def __init__(self, graph: 'PortfolioGraph'):
        """
        Initialize the weight calculation service.
        
        Parameters
        ----------
        graph : PortfolioGraph
            The portfolio graph to create weight calculations for
        """
        self.graph = graph
        self._cached_aggregators: Dict[str, WeightPathAggregator] = {}
        self._last_graph_hash: Optional[str] = None
    
    def get_aggregator(self) -> WeightPathAggregator:
        """
        Get a WeightPathAggregator instance for the current graph state.
        
        Returns a cached instance if the graph hasn't changed, otherwise
        creates a new one and caches it.
        
        Returns
        -------
        WeightPathAggregator
            Pre-populated aggregator instance
        """
        current_hash = self._compute_graph_hash()
        
        if current_hash != self._last_graph_hash or current_hash not in self._cached_aggregators:
            # Create new aggregator and populate it
            aggregator = WeightPathAggregator()
            self._populate_from_graph(aggregator)
            
            # Cache the new aggregator
            self._cached_aggregators.clear()  # Clear old cache
            self._cached_aggregators[current_hash] = aggregator
            self._last_graph_hash = current_hash
        
        return self._cached_aggregators[current_hash]
    
    def _compute_graph_hash(self) -> str:
        """
        Compute a hash representing the current graph state.
        
        This hash includes the graph structure (adjacency lists) and
        all component weights that might affect weight calculations.
        
        Returns
        -------
        str
            Hash string representing the current graph state
        """
        import hashlib
        
        # Collect graph structure data
        structure_data = []
        
        # Add adjacency list (sorted for consistency)
        adj_items = sorted(self.graph._adjacency_list.items())
        for parent_id, children in adj_items:
            children_sorted = sorted(children)
            structure_data.append(f"parent:{parent_id}->children:{','.join(children_sorted)}")
        
        # Add component weights from metric store
        weight_data = []
        for component_id in sorted(self.graph.components.keys()):
            component = self.graph.components[component_id]
            
            # Get weights for different weight types
            for weight_type in ['portfolio_weight', 'benchmark_weight', 'weight']:
                weight_metric = self.graph.metric_store.get_metric(component_id, weight_type)
                if weight_metric:
                    weight_value = weight_metric.value()
                    weight_data.append(f"{component_id}:{weight_type}={weight_value}")
        
        # Combine all data and hash
        all_data = "|".join(structure_data + sorted(weight_data))
        return hashlib.md5(all_data.encode()).hexdigest()
    
    def _populate_from_graph(self, aggregator: WeightPathAggregator) -> None:
        """
        Populate a WeightPathAggregator with data from the PortfolioGraph.
        
        Parameters
        ----------
        aggregator : WeightPathAggregator
            The aggregator instance to populate
        """
        # Add all parent-child relationships
        for parent_id, children in self.graph._adjacency_list.items():
            for child_id in children:
                aggregator.add_node_relationship(parent_id, child_id)
        
        # Set weights and mark leaves
        for component_id, component in self.graph.components.items():
            # Set weights for different weight types
            for weight_type in ['portfolio', 'benchmark']:
                weight = self._get_component_weight(component, f'{weight_type}_weight')
                aggregator.set_node_weight(component_id, weight_type, weight)
            
            # Mark leaf nodes
            if component.is_leaf():
                aggregator.mark_as_leaf(component_id)
        
        # Calculate path matrices
        aggregator.calculate_path_matrices()
    
    def _get_component_weight(self, component: 'PortfolioComponent', weight_metric: str) -> float:
        """
        Get weight for a component, falling back to base weight metric.
        
        Parameters
        ----------
        component : PortfolioComponent
            The component to get weight for
        weight_metric : str
            The weight metric name to look for
        
        Returns
        -------
        float
            The component weight, defaulting to 1.0 if not found
        """
        if not self.graph.metric_store:
            return 1.0
        
        # Try specific weight metric first
        metric = self.graph.metric_store.get_metric(component.component_id, weight_metric)
        if metric:
            return metric.value()
        
        # Fall back to base weight metric
        base_weight_metric = weight_metric.replace('portfolio_', '').replace('benchmark_', '').replace('active_', '')
        metric = self.graph.metric_store.get_metric(component.component_id, base_weight_metric)
        if metric:
            return metric.value()
        
        return 1.0
    
    def invalidate_cache(self) -> None:
        """
        Manually invalidate the cache.
        
        This can be called when you know the graph has changed
        but want to force cache invalidation.
        """
        self._cached_aggregators.clear()
        self._last_graph_hash = None


class ValidationIssue:
    """
    Represents a validation issue found during portfolio construction.
    """
    
    def __init__(self, 
                 issue_type: str, 
                 message: str, 
                 path: Optional[str] = None,
                 side: Optional[str] = None,
                 severity: str = "error"):
        """
        Initialize a validation issue.
        
        Parameters
        ----------
        issue_type : str
            Type of validation issue (e.g., "duplicate_id", "weight_mismatch")
        message : str
            Descriptive message about the issue
        path : str, optional
            Path where the issue occurred
        side : str, optional  
            Side where the issue occurred ("portfolio", "benchmark", "active")
        severity : str, default "error"
            Severity level ("error", "warning", "info")
        """
        self.issue_type = issue_type
        self.message = message
        self.path = path
        self.side = side
        self.severity = severity
    
    def __str__(self) -> str:
        """String representation of the validation issue."""
        parts = [f"[{self.severity.upper()}]"]
        if self.path:
            parts.append(f"Path: {self.path}")
        if self.side:
            parts.append(f"Side: {self.side}")
        parts.append(f"Type: {self.issue_type}")
        parts.append(f"Message: {self.message}")
        return " | ".join(parts)
    
    def __repr__(self) -> str:
        return f"ValidationIssue(type='{self.issue_type}', message='{self.message}', path='{self.path}', side='{self.side}', severity='{self.severity}')"


class WeightPathAggregatorSum:
    """
    Sum-based weight aggregator for handling absolute weights in portfolio hierarchies.
    
    This aggregator uses additive roll-up where parent weights are computed as the sum
    of children's absolute weights. It supports leverage and short positions naturally
    since weights need not sum to 1.0.
    """
    
    def __init__(self):
        """Initialize the sum-based weight path aggregator."""
        # Storage for hierarchical relationships
        self._parent_child_map: Dict[str, list] = {}  # parent_id -> [child_ids]
        self._child_parent_map: Dict[str, str] = {}   # child_id -> parent_id
        self._node_weights: Dict[str, Dict[str, float]] = {}  # node_id -> {type: weight}
        self._leaf_nodes: set = set()
        
        # Computed effective weights (direct sum from children)
        self._effective_weights: Dict[str, Dict[str, np.ndarray]] = {}  # node_id -> {type: weights_array}
        self._descendant_leaves: Dict[str, List[str]] = {}  # node_id -> [leaf_ids]
        
    def add_node_relationship(self, parent_id: str, child_id: str) -> None:
        """Add a parent-child relationship to the hierarchy."""
        if parent_id not in self._parent_child_map:
            self._parent_child_map[parent_id] = []
        self._parent_child_map[parent_id].append(child_id)
        self._child_parent_map[child_id] = parent_id
    
    def set_node_weight(self, node_id: str, weight_type: str, weight: float) -> None:
        """Set the weight for a node for a specific weight type (portfolio, benchmark)."""
        if node_id not in self._node_weights:
            self._node_weights[node_id] = {}
        self._node_weights[node_id][weight_type] = weight
    
    def mark_as_leaf(self, node_id: str) -> None:
        """Mark a node as a leaf node."""
        self._leaf_nodes.add(node_id)
    
    def calculate_path_matrices(self) -> None:
        """Calculate effective weights for all nodes using sum aggregation."""
        # Find all nodes that have descendants (intermediate and root nodes)
        nodes_with_children = set(self._parent_child_map.keys())
        
        for node_id in nodes_with_children:
            # Get all descendant leaves for this node
            descendant_leaves = self._get_descendant_leaves(node_id)
            self._descendant_leaves[node_id] = descendant_leaves
            
            # Calculate effective weights for each weight type using sum aggregation
            self._effective_weights[node_id] = {}
            for weight_type in ['portfolio', 'benchmark']:
                self._effective_weights[node_id][weight_type] = self._calculate_sum_weights(
                    node_id, descendant_leaves, weight_type
                )
    
    def _get_descendant_leaves(self, node_id: str) -> List[str]:
        """Get all descendant leaf nodes for a given node."""
        descendants = []
        
        def _traverse(current_node):
            if current_node in self._leaf_nodes:
                descendants.append(current_node)
            elif current_node in self._parent_child_map:
                for child in self._parent_child_map[current_node]:
                    _traverse(child)
        
        _traverse(node_id)
        return descendants
    
    def _calculate_sum_weights(self, node_id: str, leaf_nodes: List[str], weight_type: str) -> np.ndarray:
        """
        Calculate effective weights using sum aggregation.
        
        For sum aggregation, we simply collect the absolute weights of all descendant leaves.
        """
        weights = []
        
        for leaf_id in leaf_nodes:
            leaf_weight = self._node_weights.get(leaf_id, {}).get(weight_type, 0.0)
            weights.append(leaf_weight)
        
        return np.array(weights)
    
    def get_effective_weights(self, node_id: str, weight_type: str) -> Optional[np.ndarray]:
        """Get effective weights as numpy array for a node's descendants."""
        return self._effective_weights.get(node_id, {}).get(weight_type)
    
    def get_path_matrix(self, node_id: str, weight_type: str) -> Optional[Dict[str, float]]:
        """
        Get the path matrix for a node and weight type (compatibility method).
        
        For sum aggregation, this returns a simple mapping of leaf_id -> leaf_weight.
        """
        if node_id not in self._descendant_leaves:
            return None
        
        descendant_leaves = self._descendant_leaves[node_id]
        effective_weights = self.get_effective_weights(node_id, weight_type)
        
        if effective_weights is None:
            return None
        
        return {leaf_id: weight for leaf_id, weight in zip(descendant_leaves, effective_weights)}
    
    def normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Normalize weights to sum to 1.0.
        
        Note: For sum aggregation with leverage/shorts, normalization may not be desired.
        This method is provided for compatibility but should be used cautiously.
        """
        total = np.sum(np.abs(weights))  # Use absolute values for normalization
        if total > 0:
            return weights / total
        else:
            # Equal weights if all are zero
            return np.ones_like(weights) / len(weights) if len(weights) > 0 else weights
    
    def get_descendant_leaves(self, node_id: str) -> List[str]:
        """Public method to get descendant leaves for a node."""
        return self._descendant_leaves.get(node_id, [])
    
    def get_net_weight(self, node_id: str, weight_type: str) -> float:
        """Calculate net weight (sum of signed weights) for a node."""
        effective_weights = self.get_effective_weights(node_id, weight_type)
        if effective_weights is not None:
            return float(np.sum(effective_weights))
        return 0.0
    
    def get_gross_weight(self, node_id: str, weight_type: str) -> float:
        """Calculate gross weight (sum of absolute weights) for a node."""
        effective_weights = self.get_effective_weights(node_id, weight_type)
        if effective_weights is not None:
            return float(np.sum(np.abs(effective_weights)))
        return 0.0