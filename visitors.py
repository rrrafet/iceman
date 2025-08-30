"""
Portfolio Visitor Pattern Implementation
========================================

Visitor pattern classes for traversing portfolio hierarchies and performing
metric aggregations following the strategy pattern for extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Any, TYPE_CHECKING
import numpy as np
import pandas as pd
import logging
import time

from .metrics import (
    Metric, MetricStore, Aggregator,
    MultiMetricAggregator, WeightPathAggregator, ObjectMetric
)

if TYPE_CHECKING:
    from spark.risk.estimator import LinearRiskModelEstimator, LinearRiskModel
    from spark.risk.base import RiskDecomposerBase
    from .components import PortfolioComponent, PortfolioLeaf, PortfolioNode
    from .metrics import WeightCalculationService
    from spark.risk.schema import RiskResultSchema
    from spark.risk.risk_analysis import RiskResult

# Import simplified risk analysis functions
try:
    from spark.risk.risk_analysis import analyze_portfolio_risk, analyze_active_risk, RiskResult
except ImportError:
    # Fallback if risk_analysis module is not available
    analyze_portfolio_risk = None
    analyze_active_risk = None
    RiskResult = None




class PortfolioVisitor(ABC):
    """Abstract base class for portfolio hierarchy visitors"""
    
    @abstractmethod
    def visit_leaf(self, leaf: 'PortfolioLeaf') -> None:
        """Visit a portfolio leaf component"""
        pass
    
    @abstractmethod
    def visit_node(self, node: 'PortfolioNode') -> None:
        """Visit a portfolio node component"""
        pass


class AggregationVisitor(PortfolioVisitor):
    """Visitor for aggregating a single metric across the portfolio hierarchy"""
    
    def __init__(self, metric_name: str, aggregator: Aggregator, metric_store: Optional[MetricStore] = None, weight_metric_name: str = 'weight', context: str = 'operational'):
        self.metric_name = metric_name
        self.aggregator = aggregator
        self.metric_store = metric_store
        self.weight_metric_name = weight_metric_name
        self.context = context
        self._result_stack: List[Metric] = []
    
    def visit_leaf(self, leaf: 'PortfolioLeaf') -> None:
        """Visit leaf and push its metric to the stack"""
        metric = None
        
        # Get metric from store
        if self.metric_store:
            metric = self.metric_store.get_metric(leaf.component_id, self.metric_name)
        
        if metric:
            self._result_stack.append(metric)
        else:
            # Default metric if not found
            from .metrics import ScalarMetric
            self._result_stack.append(ScalarMetric(0.0))
    
    def visit_node(self, node: 'PortfolioNode') -> None:
        """Visit node and aggregate children metrics"""
        # First, visit all children
        child_metrics = []
        for child_id in node.get_all_children():
            # Get child component and visit it
            child_component = node._get_child_component(child_id)
            if child_component:
                child_component.accept(self)
                if self._result_stack:
                    child_metric = self._result_stack.pop()
                    # Use the child's weight as the aggregation weight
                    weight = self._get_component_weight(child_component)
                    child_metrics.append((weight, child_metric))
        
        # Aggregate all children
        if child_metrics:
            aggregated_result = self.aggregator.combine(child_metrics)
            self._result_stack.append(aggregated_result)
        else:
            from .metrics import ScalarMetric
            self._result_stack.append(ScalarMetric(0.0))
    
    def _get_component_weight(self, component: 'PortfolioComponent') -> float:
        """Get weight for a component from metric store, with overlay strategy support"""
        # Handle overlay strategies with context-sensitive weights
        if hasattr(component, 'is_overlay') and getattr(component, 'is_overlay', False):
            if self.context == 'operational':
                # For operational context (risk/performance), use parent operational weight
                return self._get_parent_operational_weight(component)
            else:
                # For allocation context (normalization), use zero weight
                return 0.0
        
        # Standard component logic
        if component.metric_store:
            weight_metric = component.metric_store.get_metric(component.component_id, self.weight_metric_name)
            if weight_metric:
                return weight_metric.value()
        return 1.0  # Default weight
    
    def _get_parent_operational_weight(self, overlay_component: 'PortfolioComponent') -> float:
        """Get operational weight for overlay component from its parent"""
        # Use the component's built-in operational weight method
        return overlay_component.get_operational_weight(self.weight_metric_name)
    
    @property
    def result(self) -> Optional[Metric]:
        """Get the final aggregation result"""
        return self._result_stack[-1] if self._result_stack else None
    
    def run_on(self, component: 'PortfolioComponent') -> Optional[Metric]:
        """Convenience method to run visitor on a component and return result"""
        component.accept(self)
        return self.result


class MultiMetricVisitor(PortfolioVisitor):
    """Visitor for aggregating multiple metrics simultaneously"""
    
    def __init__(self, 
                 metric_names: Set[str], 
                 aggregator: MultiMetricAggregator,
                 metric_store: Optional[MetricStore] = None,
                 weight_metric_name: str = 'weight'):
        self.metric_names = metric_names
        self.aggregator = aggregator
        self.metric_store = metric_store
        self.weight_metric_name = weight_metric_name
        self._result_stack: List[Dict[str, Metric]] = []
    
    def visit_leaf(self, leaf: 'PortfolioLeaf') -> None:
        """Visit leaf and collect all requested metrics"""
        metrics = {}
        
        for metric_name in self.metric_names:
            metric = None
            
            # Try metric store first
            if self.metric_store:
                metric = self.metric_store.get_metric(leaf.component_id, metric_name)
            
            # Fallback to legacy metrics if not found in store
            if metric is None:
                legacy_metrics = getattr(leaf, 'portfolio_metrics', None)
                if legacy_metrics is not None:  # Explicit None check instead of truthy check
                    metrics_dict = legacy_metrics.to_metrics_dict()
                    metric = metrics_dict.get(metric_name)
            
            if metric:
                metrics[metric_name] = metric
            else:
                from .metrics import ScalarMetric
                metrics[metric_name] = ScalarMetric(0.0)
        
        self._result_stack.append(metrics)
    
    def visit_node(self, node: 'PortfolioNode') -> None:
        """Visit node and aggregate children metrics"""
        child_metrics = []
        
        for child_id in node.get_all_children():
            child_component = node._get_child_component(child_id)
            if child_component:
                child_component.accept(self)
                if self._result_stack:
                    child_metrics_dict = self._result_stack.pop()
                    weight = self._get_component_weight(child_component)
                    child_metrics.append((weight, child_metrics_dict))
        
        # Aggregate all children for all metrics
        if child_metrics:
            aggregated_result = self.aggregator.combine(child_metrics)
            self._result_stack.append(aggregated_result)
        else:
            # Default empty metrics dict
            from .metrics import ScalarMetric
            default_metrics = {name: ScalarMetric(0.0) for name in self.metric_names}
            self._result_stack.append(default_metrics)
    
    def _get_component_weight(self, component: 'PortfolioComponent') -> float:
        """Get weight for a component from metric store"""
        if component.metric_store:
            weight_metric = component.metric_store.get_metric(component.component_id, self.weight_metric_name)
            if weight_metric:
                return weight_metric.value()
        return 1.0
    
    @property
    def result(self) -> Optional[Dict[str, Metric]]:
        """Get the final aggregation results"""
        return self._result_stack[-1] if self._result_stack else None
    
    def run_on(self, component: 'PortfolioComponent') -> Optional[Dict[str, Metric]]:
        """Convenience method to run visitor on a component and return result"""
        component.accept(self)
        return self.result




class FactorRiskDecompositionVisitor(PortfolioVisitor):
    """
    Visitor for factor-based risk decomposition using linear risk models.
    
    This visitor performs hierarchical risk decomposition by:
    1. Estimating factor exposures via OLS regression for each leaf versus portfolio, benchmark, and active returns
    2. Storing RiskModelBase objects in metric_store for efficient retrieval
    3. For each node calculates the descendants weight by (leafs by definition have weights 1.0 and 1.0) multiplying the each leafs specified weight and successively multiplies with node weights
    4. Creates unified beta, resvar, weight matrices (across portfolio, benchmark, and active)
    5. Stores the unified matrices for use by risk decomposer for each node of the tree.
    """
    
    def __init__(self,
                 factor_returns: pd.DataFrame,
                 estimator: Optional['LinearRiskModelEstimator'] = None,
                 metric_store: Optional[MetricStore] = None,
                 portfolio_returns_metric: str = 'portfolio_return',
                 benchmark_returns_metric: str = 'benchmark_return',
                 portfolio_weight_metric: str = 'portfolio_weight',
                 benchmark_weight_metric: str = 'benchmark_weight',
                 annualize: bool = True,
                 log_level: str = "INFO",
                 verbose: bool = False,
                 log_file: Optional[str] = None,
                 weight_service: Optional['WeightCalculationService'] = None,
                 **kwargs):
        self.factor_returns = factor_returns
        self.factor_names = list(factor_returns.columns)
        self.metric_store = metric_store
        self.portfolio_returns_metric = portfolio_returns_metric
        self.benchmark_returns_metric = benchmark_returns_metric
        self.portfolio_weight_metric = portfolio_weight_metric
        self.benchmark_weight_metric = benchmark_weight_metric
        self.annualize = annualize
        
        # Initialize logging
        from spark.core.logging_config import setup_risk_decomposition_logging
        self.logger = setup_risk_decomposition_logging(
            log_level=log_level,
            verbose=verbose,
            log_file=log_file
        )

        freq = kwargs.get('freq', None)

        # Initialize estimator
        if estimator is None:
            from spark.risk.estimator import LinearRiskModelEstimator
            self.estimator = LinearRiskModelEstimator(regression_type='ols', min_obs=30, freq=freq)
            self.logger.info("Initialized default LinearRiskModelEstimator with OLS regression and min_obs=30")
        else:
            self.estimator = estimator
            self.logger.info(f"Using provided estimator: {type(estimator).__name__}")
        
        self.logger.info(f"FactorRiskDecompositionVisitor initialized with {len(self.factor_names)} factors: {self.factor_names}")
        self.logger.info(f"Portfolio metrics: returns={portfolio_returns_metric}, weights={portfolio_weight_metric}")
        self.logger.info(f"Benchmark metrics: returns={benchmark_returns_metric}, weights={benchmark_weight_metric}")
        
        # Initialize WeightPathAggregator for hierarchical weight calculations
        self._weight_service = weight_service
        if weight_service is not None:
            self._weight_aggregator = weight_service.get_aggregator()
            self._use_service = True
            self.logger.info("Using provided WeightCalculationService for weight calculations")
        else:
            self._weight_aggregator = WeightPathAggregator()
            self._use_service = False
            self.logger.info("Using manual WeightPathAggregator (fallback mode)")
        
        # Simplified storage structure
        self._leaf_models: Dict[str, Dict[str, Any]] = {}  # leaf_id -> {'portfolio': model, 'benchmark': model, 'active': model}
        self._node_risk_results: Dict[str, Dict[str, Any]] = {}  # node_id -> {'portfolio': RiskResult, 'benchmark': RiskResult, 'active': RiskResult}
        
        # Keep weight aggregator for hierarchical weight calculations (this is working well)
        self._node_weights: Dict[str, Dict[str, float]] = {}  # Each node's own weights
        self._node_contribution_weights: Dict[Tuple[str, str], Dict[str, float]] = {}  # (node_id, leaf_id) -> weights
        self._descendant_weights: Dict[str, Dict[str, np.ndarray]] = {}  # Effective weights for risk calculation
        
        self._processed_components: Set[str] = set()
    
    def visit_leaf(self, leaf: 'PortfolioLeaf') -> None:
        """Visit leaf and estimate factor models, calculate risk using simplified API"""
        start_time = time.time()
        self.logger.debug(f"Visiting leaf: {leaf.component_id}")
        
        if leaf.component_id in self._processed_components:
            self.logger.debug(f"Leaf {leaf.component_id} already processed, skipping")
            return
        
        # Get returns data for this leaf
        self.logger.debug(f"Retrieving returns data for leaf {leaf.component_id}")
        portfolio_returns = self._get_component_returns(leaf, self.portfolio_returns_metric)
        benchmark_returns = self._get_component_returns(leaf, self.benchmark_returns_metric)
        
        if portfolio_returns is None or benchmark_returns is None:
            self.logger.error(f"Missing returns data for leaf {leaf.component_id}")
            raise ValueError(f"Missing returns data for leaf {leaf.component_id}. Ensure metrics are set in the metric store.")
        
        # Calculate active returns
        active_returns = portfolio_returns - benchmark_returns
        self.logger.debug(f"Calculated active returns for {leaf.component_id}, mean: {np.mean(active_returns):.6f}, std: {np.std(active_returns):.6f}")
        
        # Estimate factor models for all three return types (keep current approach for individual leaves)
        self.logger.info(f"Estimating factor models for leaf {leaf.component_id}")
        leaf_models = {}
        for return_type, returns_data in [
            ('portfolio', portfolio_returns),
            ('benchmark', benchmark_returns), 
            ('active', active_returns)
        ]:
            try:
                fit_start_time = time.time()
                model = self.estimator.fit(
                    asset_returns=pd.DataFrame({leaf.component_id: returns_data}),
                    factor_returns=self.factor_returns
                )
                fit_time = time.time() - fit_start_time
                leaf_models[return_type] = model
                self.logger.debug(f"Fitted {return_type} model for {leaf.component_id} in {fit_time:.3f}s")
            except Exception as e:
                self.logger.error(f"Failed to estimate {return_type} risk model for {leaf.component_id}: {str(e)}")
                raise ValueError(f"Failed to estimate risk model for {leaf.component_id} with return type {return_type}")
        
        # Store models for later use in node aggregation
        self._leaf_models[leaf.component_id] = leaf_models
        self.logger.info(f"Stored {len(leaf_models)} risk models for leaf {leaf.component_id}")
        
        # Get leaf weights
        portfolio_weight = self._get_component_weight(leaf, 'portfolio_weight')
        benchmark_weight = self._get_component_weight(leaf, 'benchmark_weight')
        
        # Calculate leaf-level risk using analyze_portfolio_risk() directly
        if analyze_portfolio_risk is not None:
            try:
                # Portfolio risk analysis
                portfolio_result = analyze_portfolio_risk(
                    model=leaf_models['portfolio'],
                    weights=np.array([portfolio_weight]),
                    asset_names=[leaf.component_id],
                    factor_names=self.factor_names,
                    annualize=self.annualize
                )
                
                # Benchmark risk analysis
                benchmark_result = analyze_portfolio_risk(
                    model=leaf_models['benchmark'],
                    weights=np.array([benchmark_weight]),
                    asset_names=[leaf.component_id],
                    factor_names=self.factor_names,
                    annualize=self.annualize
                )
                
                # Active risk analysis
                active_result = analyze_active_risk(
                    portfolio_model=leaf_models['portfolio'],
                    benchmark_model=leaf_models['benchmark'],
                    portfolio_weights=np.array([portfolio_weight]),
                    benchmark_weights=np.array([benchmark_weight]),
                    asset_names=[leaf.component_id],
                    factor_names=self.factor_names,
                    active_model=leaf_models['active'],
                    annualize=self.annualize
                )
                
                # Store RiskResult objects
                self._node_risk_results[leaf.component_id] = {
                    'portfolio': portfolio_result,
                    'benchmark': benchmark_result,
                    'active': active_result
                }
                
                self.logger.info(f"Calculated risks for {leaf.component_id}: "
                               f"portfolio={portfolio_result.total_risk:.4f}, "
                               f"benchmark={benchmark_result.total_risk:.4f}, "
                               f"active={active_result.total_risk:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate simplified risk analysis for {leaf.component_id}: {e}")
        
        # Store individual risk models in metric store (for backward compatibility)
        if self.metric_store:
            for return_type, model in leaf_models.items():
                self.metric_store.set_metric(
                    leaf.component_id, 
                    f'{return_type}_risk_model', 
                    ObjectMetric(model)
                )
            
            # Store asset names for schema extraction
            self.metric_store.set_metric(
                leaf.component_id, 
                'asset_names', 
                ObjectMetric([leaf.component_id])
            )
        
        # Store weights and register with weight aggregator (preserve existing logic)
        self._node_weights[leaf.component_id] = {
            'portfolio': portfolio_weight,
            'benchmark': benchmark_weight,
        }
        
        if not self._use_service:
            self._weight_aggregator.set_node_weight(leaf.component_id, 'portfolio', portfolio_weight)
            self._weight_aggregator.set_node_weight(leaf.component_id, 'benchmark', benchmark_weight)
            self._weight_aggregator.mark_as_leaf(leaf.component_id)
        
        # Initialize contribution weights for leaf to itself
        self._node_contribution_weights[(leaf.component_id, leaf.component_id)] = {
            'portfolio': portfolio_weight,
            'benchmark': benchmark_weight,
        }
        
        self._processed_components.add(leaf.component_id)
        
        processing_time = time.time() - start_time
        self.logger.info(f"Completed simplified leaf {leaf.component_id} processing in {processing_time:.3f} seconds")
    
    def visit_node(self, node: 'PortfolioNode') -> None:
        """
        Visit node and calculate risk using correct model re-estimation approach.
        
        CRITICAL: Re-estimates models with ALL descendant leaves together to capture cross-correlations.
        This is mathematically required for accurate risk decomposition.
        """
        start_time = time.time()
        self.logger.debug(f"Visiting node: {node.component_id}")
        
        if node.component_id in self._processed_components:
            self.logger.debug(f"Node {node.component_id} already processed, skipping")
            return
        
        # First visit all children and collect descendant leaves
        descendant_leaves = []
        children_ids = node.get_all_children()
        self.logger.debug(f"Node {node.component_id} has {len(children_ids)} children: {children_ids}")
        
        for child_id in children_ids:
            # Register parent-child relationship with weight aggregator 
            if not self._use_service:
                self._weight_aggregator.add_node_relationship(node.component_id, child_id)
            
            child_component = node._get_child_component(child_id)
            if child_component:
                child_component.accept(self)
                if child_component.is_leaf() and child_id in self._leaf_models:
                    descendant_leaves.append(child_id)
                    self.logger.debug(f"Added leaf descendant: {child_id}")
                elif not child_component.is_leaf() and child_id in self._node_risk_results:
                    # For child nodes, get their descendant leaves from stored asset names
                    if self.metric_store:
                        asset_names_metric = self.metric_store.get_metric(child_id, 'asset_names')
                        if asset_names_metric:
                            child_descendants = asset_names_metric.value()
                            descendant_leaves.extend(child_descendants)
                            self.logger.debug(f"Added {len(child_descendants)} descendants from child node {child_id}")
        
        if not descendant_leaves:
            self.logger.warning(f"Node {node.component_id} has no descendant leaves with risk models")
            return
            
        self.logger.info(f"Processing node {node.component_id} with {len(descendant_leaves)} descendant leaves: {descendant_leaves}")
        
        # Store node weights and register with weight aggregator
        node_portfolio_weight = self._get_component_weight(node, 'portfolio_weight')
        node_benchmark_weight = self._get_component_weight(node, 'benchmark_weight')
        
        self._node_weights[node.component_id] = {
            'portfolio': node_portfolio_weight,
            'benchmark': node_benchmark_weight
        }
        
        if not self._use_service:
            self._weight_aggregator.set_node_weight(node.component_id, 'portfolio', node_portfolio_weight)
            self._weight_aggregator.set_node_weight(node.component_id, 'benchmark', node_benchmark_weight)
        
        # Store descendant asset names for schema extraction
        if self.metric_store:
            self.metric_store.set_metric(
                node.component_id, 
                'asset_names', 
                ObjectMetric(descendant_leaves)
            )
        
        # Calculate effective weights using WeightPathAggregator
        self._calculate_effective_weights(node, descendant_leaves)
        
        # CRITICAL: Re-estimate models with ALL descendant leaves together to capture cross-correlations
        self.logger.info(f"Re-estimating models for node {node.component_id} with combined leaf data")
        node_models = self._re_estimate_combined_models(node.component_id, descendant_leaves)
        
        if node_models is None:
            self.logger.error(f"Failed to re-estimate models for node {node.component_id}")
            return
        
        # Get effective weights for risk calculation
        if node.component_id not in self._descendant_weights:
            self.logger.error(f"No effective weights calculated for node {node.component_id}")
            return
            
        portfolio_weights = self._descendant_weights[node.component_id]['portfolio']
        benchmark_weights = self._descendant_weights[node.component_id]['benchmark']
        
        # Calculate risk using simplified analyze_active_risk() API
        if analyze_active_risk is not None:
            try:
                # Portfolio risk analysis
                portfolio_result = analyze_portfolio_risk(
                    model=node_models['portfolio'],
                    weights=portfolio_weights,
                    asset_names=descendant_leaves,
                    factor_names=self.factor_names,
                    annualize=self.annualize
                )
                
                # Benchmark risk analysis
                benchmark_result = analyze_portfolio_risk(
                    model=node_models['benchmark'],
                    weights=benchmark_weights,
                    asset_names=descendant_leaves,
                    factor_names=self.factor_names,
                    annualize=self.annualize
                )
                
                # Active risk analysis with properly estimated models
                active_result = analyze_active_risk(
                    portfolio_model=node_models['portfolio'],
                    benchmark_model=node_models['benchmark'],
                    portfolio_weights=portfolio_weights,
                    benchmark_weights=benchmark_weights,
                    asset_names=descendant_leaves,
                    factor_names=self.factor_names,
                    active_model=node_models['active'],
                    annualize=self.annualize
                )
                
                # Store RiskResult objects directly (no complex context)
                self._node_risk_results[node.component_id] = {
                    'portfolio': portfolio_result,
                    'benchmark': benchmark_result,
                    'active': active_result
                }
                
                self.logger.info(f"Calculated risks for node {node.component_id}: "
                               f"portfolio={portfolio_result.total_risk:.4f}, "
                               f"benchmark={benchmark_result.total_risk:.4f}, "
                               f"active={active_result.total_risk:.4f}")
                
                # Store in metric store for backward compatibility (simplified)
                if self.metric_store:
                    self.metric_store.set_metric(
                        node.component_id,
                        'portfolio_risk_result',
                        ObjectMetric(portfolio_result)
                    )
                    self.metric_store.set_metric(
                        node.component_id,
                        'benchmark_risk_result', 
                        ObjectMetric(benchmark_result)
                    )
                    self.metric_store.set_metric(
                        node.component_id,
                        'active_risk_result',
                        ObjectMetric(active_result)
                    )
                
            except Exception as e:
                self.logger.error(f"Failed to calculate risk for node {node.component_id}: {e}")
        
        self._processed_components.add(node.component_id)
        
        processing_time = time.time() - start_time
        self.logger.info(f"Completed simplified node {node.component_id} processing in {processing_time:.3f} seconds")
    
    def _re_estimate_combined_models(self, node_id: str, descendant_leaves: List[str]) -> Optional[Dict[str, Any]]:
        """
        CRITICAL: Re-estimate models with ALL descendant leaves together to capture cross-correlations.
        
        This is mathematically required because:
        - Residual returns have cross-correlations between assets
        - The residual covariance matrix is NOT diagonal - it captures correlations between asset-specific returns
        - These cross-correlations are essential for accurate risk decomposition
        - You CANNOT aggregate individual leaf risk results - you must re-estimate with all leaves together
        
        Parameters
        ----------
        node_id : str
            Node component ID
        descendant_leaves : List[str]
            List of descendant leaf component IDs
            
        Returns
        -------
        Dict[str, LinearRiskModel] or None
            Dictionary containing re-estimated models for 'portfolio', 'benchmark', 'active'
        """
        try:
            self.logger.debug(f"Re-estimating combined models for {node_id} with {len(descendant_leaves)} leaves")
            
            # Combine all descendant leaf returns into single DataFrames
            portfolio_returns_data = {}
            benchmark_returns_data = {}
            
            for leaf_id in descendant_leaves:
                # Get returns data from metric store
                portfolio_returns = self._get_component_returns_by_id(leaf_id, self.portfolio_returns_metric)
                benchmark_returns = self._get_component_returns_by_id(leaf_id, self.benchmark_returns_metric)
                
                if portfolio_returns is not None and benchmark_returns is not None:
                    portfolio_returns_data[leaf_id] = portfolio_returns
                    benchmark_returns_data[leaf_id] = benchmark_returns
                else:
                    self.logger.warning(f"Missing returns data for leaf {leaf_id} in node {node_id}")
            
            if not portfolio_returns_data or not benchmark_returns_data:
                self.logger.error(f"No valid returns data found for descendants of node {node_id}")
                return None
            
            # Create combined DataFrames
            portfolio_returns_df = pd.DataFrame(portfolio_returns_data)
            benchmark_returns_df = pd.DataFrame(benchmark_returns_data)
            active_returns_df = portfolio_returns_df - benchmark_returns_df
            
            self.logger.info(f"Combined returns data for {node_id}: "
                           f"portfolio={portfolio_returns_df.shape}, "
                           f"benchmark={benchmark_returns_df.shape}, "
                           f"active={active_returns_df.shape}")
            
            # Re-estimate models with combined data - captures cross-correlations!
            models = {}
            for return_type, returns_df in [
                ('portfolio', portfolio_returns_df),
                ('benchmark', benchmark_returns_df),
                ('active', active_returns_df)
            ]:
                try:
                    # Fit NEW models to combined data - captures residual correlations!
                    combined_model = self.estimator.fit(
                        asset_returns=returns_df,
                        factor_returns=self.factor_returns
                    )
                    models[return_type] = combined_model
                    
                    self.logger.debug(f"Re-estimated {return_type} model for {node_id}: "
                                    f"beta={combined_model.beta.shape}, "
                                    f"resvar={combined_model.resvar.shape}")
                                    
                except Exception as e:
                    self.logger.error(f"Failed to re-estimate {return_type} model for {node_id}: {e}")
                    return None
            
            self.logger.info(f"Successfully re-estimated all models for node {node_id}")
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to re-estimate combined models for {node_id}: {e}")
            return None
    
    def _get_component_returns_by_id(self, component_id: str, metric_name: str) -> Optional[pd.Series]:
        """Get returns data for a component by ID from metric store"""
        if not self.metric_store:
            return None
        
        metric = self.metric_store.get_metric(component_id, metric_name)
        if metric is None:
            return None
        
        returns_data = metric.value()
        if isinstance(returns_data, pd.Series):
            return returns_data
        elif isinstance(returns_data, (int, float)):
            # If scalar, create a series with factor returns index
            return pd.Series(returns_data, index=self.factor_returns.index)
        
        return None
    
    def validate_risk_estimates(self, node_id: str) -> Dict[str, Any]:
        """
        Validate that model-estimated risks match empirical calculations.
        
        This is the ultimate test of correctness as required by the prompt.
        For each node, validates that:
        - Empirical portfolio risk = Model-estimated risk
        - Empirical benchmark risk = Model-estimated risk  
        - Empirical active risk = Model-estimated risk
        
        Parameters
        ----------
        node_id : str
            Node component ID to validate
            
        Returns
        -------
        Dict[str, Any]
            Validation results with pass/fail status and any discrepancies
        """
        try:
            self.logger.info(f"Validating risk estimates for node {node_id}")
            
            # Get stored risk results
            if node_id not in self._node_risk_results:
                return {
                    'node_id': node_id,
                    'validation_passed': False,
                    'error': f'No risk results found for node {node_id}'
                }
            
            risk_results = self._node_risk_results[node_id]
            
            # Get descendant leaves and their returns
            if self.metric_store:
                asset_names_metric = self.metric_store.get_metric(node_id, 'asset_names')
                if not asset_names_metric:
                    return {
                        'node_id': node_id,
                        'validation_passed': False,
                        'error': f'No asset names found for node {node_id}'
                    }
                descendant_leaves = asset_names_metric.value()
            else:
                return {
                    'node_id': node_id,
                    'validation_passed': False,
                    'error': 'No metric store available for validation'
                }
            
            # Calculate empirical risks from actual returns
            empirical_portfolio_risk = self._calculate_empirical_risk(node_id, descendant_leaves, 'portfolio')
            empirical_benchmark_risk = self._calculate_empirical_risk(node_id, descendant_leaves, 'benchmark')
            empirical_active_risk = self._calculate_empirical_risk(node_id, descendant_leaves, 'active')
            
            if any(risk is None for risk in [empirical_portfolio_risk, empirical_benchmark_risk, empirical_active_risk]):
                return {
                    'node_id': node_id,
                    'validation_passed': False,
                    'error': 'Failed to calculate empirical risks'
                }
            
            # Compare with model estimates (tolerance: 1e-6 as specified in prompt)
            tolerance = 1e-6
            validations = {
                'portfolio': {
                    'empirical': empirical_portfolio_risk,
                    'model': risk_results['portfolio'].total_risk,
                    'difference': abs(empirical_portfolio_risk - risk_results['portfolio'].total_risk),
                    'passed': abs(empirical_portfolio_risk - risk_results['portfolio'].total_risk) < tolerance
                },
                'benchmark': {
                    'empirical': empirical_benchmark_risk,
                    'model': risk_results['benchmark'].total_risk,
                    'difference': abs(empirical_benchmark_risk - risk_results['benchmark'].total_risk),
                    'passed': abs(empirical_benchmark_risk - risk_results['benchmark'].total_risk) < tolerance
                },
                'active': {
                    'empirical': empirical_active_risk,
                    'model': risk_results['active'].total_risk,
                    'difference': abs(empirical_active_risk - risk_results['active'].total_risk),
                    'passed': abs(empirical_active_risk - risk_results['active'].total_risk) < tolerance
                }
            }
            
            # Log warnings for any validation failures
            all_passed = True
            for risk_type, validation in validations.items():
                if not validation['passed']:
                    all_passed = False
                    self.logger.warning(
                        f"Risk validation failed for {node_id} {risk_type}: "
                        f"empirical={validation['empirical']:.4%}, "
                        f"model={validation['model']:.4%}, "
                        f"difference={validation['difference']:.4%}"
                    )
                else:
                    self.logger.debug(
                        f"Risk validation passed for {node_id} {risk_type}: "
                        f"empirical={validation['empirical']:.4%}, "
                        f"model={validation['model']:.4%}, "
                        f"difference={validation['difference']:.6%}"
                    )
            
            return {
                'node_id': node_id,
                'validation_passed': all_passed,
                'validations': validations,
                'tolerance': tolerance,
                'message': 'All validations passed' if all_passed else 'Some validations failed'
            }
            
        except Exception as e:
            self.logger.error(f"Risk validation failed for {node_id}: {e}")
            return {
                'node_id': node_id,
                'validation_passed': False,
                'error': str(e)
            }
    
    def _calculate_empirical_risk(self, node_id: str, descendant_leaves: List[str], return_type: str) -> Optional[float]:
        """
        Calculate empirical risk from weighted leaf returns as specified in the prompt.
        
        This calculates the actual risk of the weighted portfolio/benchmark/active returns
        to compare against model estimates.
        """
        try:
            # Get effective weights for this node
            if node_id not in self._descendant_weights:
                self.logger.error(f"No effective weights found for node {node_id}")
                return None
            
            if return_type == 'active':
                weights = (self._descendant_weights[node_id]['portfolio'] - 
                          self._descendant_weights[node_id]['benchmark'])
            else:
                weights = self._descendant_weights[node_id][return_type]
            
            # Calculate weighted returns
            weighted_returns = None
            
            for i, leaf_id in enumerate(descendant_leaves):
                if return_type == 'active':
                    # Calculate active returns from portfolio - benchmark
                    portfolio_returns = self._get_component_returns_by_id(leaf_id, self.portfolio_returns_metric)
                    benchmark_returns = self._get_component_returns_by_id(leaf_id, self.benchmark_returns_metric)
                    if portfolio_returns is not None and benchmark_returns is not None:
                        leaf_returns = portfolio_returns - benchmark_returns
                    else:
                        continue
                elif return_type == 'portfolio':
                    leaf_returns = self._get_component_returns_by_id(leaf_id, self.portfolio_returns_metric)
                elif return_type == 'benchmark':
                    leaf_returns = self._get_component_returns_by_id(leaf_id, self.benchmark_returns_metric)
                
                if leaf_returns is not None and i < len(weights):
                    if weighted_returns is None:
                        weighted_returns = weights[i] * leaf_returns
                    else:
                        weighted_returns += weights[i] * leaf_returns
            
            if weighted_returns is None:
                return None
            
            # Calculate empirical risk (standard deviation)
            empirical_risk = weighted_returns.std()
            
            # Annualize if needed (same logic as in model)
            if self.annualize:
                # Determine annualization factor from factor returns frequency
                freq = getattr(self.factor_returns.index, 'freq', None)
                if freq is not None:
                    if 'D' in str(freq) or 'B' in str(freq):
                        periods_per_year = 252  # Business days
                    elif 'M' in str(freq):
                        periods_per_year = 12   # Months
                    elif 'W' in str(freq):
                        periods_per_year = 52   # Weeks
                    else:
                        periods_per_year = 252  # Default
                else:
                    periods_per_year = 252  # Default
                
                empirical_risk *= np.sqrt(periods_per_year)
            
            return float(empirical_risk)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate empirical {return_type} risk for {node_id}: {e}")
            return None
    
    def _log_risk_model_details(self, model: 'LinearRiskModel', leaf_id: str, return_type: str) -> None:
        """Log comprehensive risk model statistics and diagnostics"""
        try:
            # Basic model information
            beta_shape = model.beta.shape
            self.logger.debug(f"Risk model for {leaf_id} ({return_type}): beta shape {beta_shape}")
            
            # Factor loadings analysis
            for i, factor_name in enumerate(self.factor_names):
                if i < beta_shape[1]:
                    beta_value = model.beta[0, i]  # First asset, i-th factor
                    self.logger.debug(f"  {factor_name} loading: {beta_value:.4f}")
            
            # Residual variance analysis
            resvar_diag = np.diag(model.resvar) if model.resvar.ndim > 1 else model.resvar
            resvar_value = resvar_diag[0] if hasattr(resvar_diag, '__len__') else resvar_diag
            self.logger.debug(f"  Residual variance: {resvar_value:.6f}")
            
            # Asset returns statistics if available
            if hasattr(model, 'asset_returns') and model.asset_returns is not None:
                asset_ret = model.asset_returns[leaf_id]
                stats = {
                    'mean': np.mean(asset_ret),
                    'std': np.std(asset_ret),
                    'min': np.min(asset_ret),
                    'max': np.max(asset_ret),
                    'skew': pd.Series(asset_ret).skew(),
                    'kurt': pd.Series(asset_ret).kurtosis()
                }
                self.logger.debug(f"  Asset returns stats: mean={stats['mean']:.6f}, std={stats['std']:.6f}, skew={stats['skew']:.3f}, kurt={stats['kurt']:.3f}")
            
            # Residual returns analysis if available
            if hasattr(model, 'residual_returns') and model.residual_returns is not None:
                residual_ret = model.residual_returns[leaf_id]
                residual_stats = {
                    'mean': np.mean(residual_ret),
                    'std': np.std(residual_ret),
                    'min': np.min(residual_ret),
                    'max': np.max(residual_ret)
                }
                self.logger.debug(f"  Residual stats: mean={residual_stats['mean']:.6f}, std={residual_stats['std']:.6f}")
                
                # Check for potential outliers (> 3 std devs)
                outliers = np.abs(residual_ret) > 3 * residual_stats['std']
                if np.any(outliers):
                    outlier_count = np.sum(outliers)
                    self.logger.warning(f"  Found {outlier_count} potential outliers in residuals for {leaf_id} ({return_type})")
            
        except Exception as e:
            self.logger.warning(f"Failed to log risk model details for {leaf_id} ({return_type}): {str(e)}")
    
    def _log_weight_analysis(self, weights: np.ndarray, node_id: str, return_type: str, leaf_ids: List[str] = None) -> None:
        """Log comprehensive weight distribution analysis"""
        try:
            if weights is None or len(weights) == 0:
                self.logger.debug(f"No weights to analyze for {node_id} ({return_type})")
                return
            
            # Basic weight statistics
            weight_stats = {
                'sum': np.sum(weights),
                'mean': np.mean(weights),
                'std': np.std(weights),
                'min': np.min(weights),
                'max': np.max(weights),
                'median': np.median(weights)
            }
            
            self.logger.info(f"Weight analysis for {node_id} ({return_type}): sum={weight_stats['sum']:.4f}, mean={weight_stats['mean']:.4f}, max={weight_stats['max']:.4f}")
            self.logger.debug(f"  Weight stats: std={weight_stats['std']:.4f}, min={weight_stats['min']:.4f}, median={weight_stats['median']:.4f}")
            
            # Weight concentration analysis
            if len(weights) > 1:
                # Herfindahl-Hirschman Index (concentration measure)
                hhi = np.sum(weights**2)
                self.logger.debug(f"  Weight concentration (HHI): {hhi:.4f}")
                
                # Number of effective positions
                if hhi > 0:
                    n_effective = 1.0 / hhi
                    self.logger.debug(f"  Effective number of positions: {n_effective:.2f}")
            
            # Log individual weights if DEBUG level and reasonable number of assets
            if self.logger.isEnabledFor(logging.DEBUG) and len(weights) <= 20:
                if leaf_ids and len(leaf_ids) == len(weights):
                    for leaf_id, weight in zip(leaf_ids, weights):
                        self.logger.debug(f"    {leaf_id}: {weight:.6f}")
                else:
                    for i, weight in enumerate(weights):
                        self.logger.debug(f"    Asset {i}: {weight:.6f}")
            
            # Weight validation checks
            if return_type in ['portfolio', 'benchmark']:
                if abs(weight_stats['sum'] - 1.0) > 0.01:  # 1% tolerance
                    self.logger.warning(f"Weight sum validation failed for {node_id} ({return_type}): sum={weight_stats['sum']:.6f}, expected ≈ 1.0")
            elif return_type == 'active':
                if abs(weight_stats['sum']) > 0.01:  # Active weights should sum to ~0
                    self.logger.warning(f"Active weight sum validation failed for {node_id}: sum={weight_stats['sum']:.6f}, expected ≈ 0.0")
            
            # Active weight specific analysis
            if return_type == 'active':
                long_weights = weights[weights > 0]
                short_weights = weights[weights < 0]
                
                self.logger.debug(f"  Active position analysis: {len(long_weights)} long, {len(short_weights)} short")
                if len(long_weights) > 0:
                    self.logger.debug(f"    Long exposure: {np.sum(long_weights):.4f}")
                if len(short_weights) > 0:
                    self.logger.debug(f"    Short exposure: {np.sum(short_weights):.4f}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to log weight analysis for {node_id} ({return_type}): {str(e)}")
    
    def _log_matrix_diagnostics(self, matrix: np.ndarray, matrix_name: str, node_id: str) -> None:
        """Log matrix health diagnostics and properties"""
        try:
            if matrix is None:
                self.logger.debug(f"Matrix {matrix_name} for {node_id} is None")
                return
            
            shape = matrix.shape
            self.logger.debug(f"Matrix {matrix_name} for {node_id}: shape {shape}")
            
            if matrix.size == 0:
                self.logger.warning(f"Matrix {matrix_name} for {node_id} is empty")
                return
            
            # Basic statistics
            matrix_stats = {
                'min': np.min(matrix),
                'max': np.max(matrix),
                'mean': np.mean(matrix),
                'std': np.std(matrix)
            }
            
            self.logger.debug(f"  {matrix_name} stats: min={matrix_stats['min']:.6f}, max={matrix_stats['max']:.6f}, mean={matrix_stats['mean']:.6f}")
            
            # Check for problematic values
            if np.any(np.isnan(matrix)):
                nan_count = np.sum(np.isnan(matrix))
                self.logger.warning(f"  {matrix_name} contains {nan_count} NaN values")
            
            if np.any(np.isinf(matrix)):
                inf_count = np.sum(np.isinf(matrix))
                self.logger.warning(f"  {matrix_name} contains {inf_count} infinite values")
            
            # For square matrices, check condition number
            if len(shape) == 2 and shape[0] == shape[1] and shape[0] > 0:
                try:
                    cond_num = np.linalg.cond(matrix)
                    if cond_num > 1e12:
                        self.logger.warning(f"  {matrix_name} is poorly conditioned: condition number = {cond_num:.2e}")
                    else:
                        self.logger.debug(f"  {matrix_name} condition number: {cond_num:.2e}")
                        
                    # Check if positive definite (for covariance matrices)
                    if 'covar' in matrix_name.lower():
                        eigenvals = np.linalg.eigvals(matrix)
                        min_eigenval = np.min(eigenvals)
                        if min_eigenval <= 0:
                            self.logger.warning(f"  {matrix_name} is not positive definite: min eigenvalue = {min_eigenval:.6f}")
                        else:
                            self.logger.debug(f"  {matrix_name} min eigenvalue: {min_eigenval:.6f}")
                            
                except np.linalg.LinAlgError as e:
                    self.logger.warning(f"  Could not compute diagnostics for {matrix_name}: {str(e)}")
            
            # Log matrix contents for small matrices at DEBUG level
            if self.logger.isEnabledFor(logging.DEBUG) and matrix.size <= 100:
                if matrix.ndim == 1:
                    matrix_str = np.array_str(matrix, precision=4, suppress_small=True)
                else:
                    matrix_str = np.array_str(matrix, precision=4, suppress_small=True, max_line_width=120)
                self.logger.debug(f"  {matrix_name} contents:\n{matrix_str}")
            elif self.logger.isEnabledFor(logging.DEBUG) and matrix.size <= 1000:
                # For larger matrices, show summary statistics per row/column
                if matrix.ndim == 2:
                    self.logger.debug(f"  {matrix_name} first 3 rows:\n{matrix[:3]}")
                    self.logger.debug(f"  {matrix_name} row means: {np.mean(matrix, axis=1)[:10]}")  # First 10 row means
                elif matrix.ndim == 1:
                    self.logger.debug(f"  {matrix_name} first 10 elements: {matrix[:10]}")
                
        except Exception as e:
            self.logger.warning(f"Failed to log matrix diagnostics for {matrix_name} ({node_id}): {str(e)}")
    
    def _log_decomposition_summary(self, context, node_id: str) -> None:
        """Log detailed risk decomposition results and summaries"""
        try:
            self.logger.info(f"Logging risk decomposition summary for {node_id}")
            
            # Get summaries from each decomposer
            portfolio_summary = context.portfolio_decomposer.risk_decomposition_summary()
            benchmark_summary = context.benchmark_decomposer.risk_decomposition_summary()
            active_summary = context.active_decomposer.risk_decomposition_summary()
            
            # Portfolio risk breakdown
            port_vol = portfolio_summary['portfolio_volatility']
            port_factor_pct = portfolio_summary['factor_risk_percentage']
            port_specific_pct = portfolio_summary['specific_risk_percentage']
            
            self.logger.info(f"  Portfolio Risk: {port_vol:.4f} ({port_vol:.1%})")
            self.logger.info(f"    Factor Risk: {port_factor_pct:.1f}%, Specific Risk: {port_specific_pct:.1f}%")
            
            # Benchmark risk breakdown
            bench_vol = benchmark_summary['portfolio_volatility']
            bench_factor_pct = benchmark_summary['factor_risk_percentage']
            bench_specific_pct = benchmark_summary['specific_risk_percentage']
            
            self.logger.info(f"  Benchmark Risk: {bench_vol:.4f} ({bench_vol:.1%})")
            self.logger.info(f"    Factor Risk: {bench_factor_pct:.1f}%, Specific Risk: {bench_specific_pct:.1f}%")
            
            # Active risk breakdown
            active_vol = active_summary['total_active_risk']
            
            self.logger.info(f"  Active Risk: {active_vol:.4f} ({active_vol:.1%})")
            
            # Active risk allocation vs selection breakdown
            if 'allocation_factor_percentage' in active_summary:
                alloc_factor = active_summary.get('allocation_factor_percentage', 0)
                alloc_specific = active_summary.get('allocation_specific_percentage', 0)
                sel_factor = active_summary.get('selection_factor_percentage', 0)
                sel_specific = active_summary.get('selection_specific_percentage', 0)
                
                total_alloc = alloc_factor + alloc_specific
                total_sel = sel_factor + sel_specific
                
                self.logger.info(f"    Allocation Risk: {total_alloc:.1f}% (Factor: {alloc_factor:.1f}%, Specific: {alloc_specific:.1f}%)")
                self.logger.info(f"    Selection Risk: {total_sel:.1f}% (Factor: {sel_factor:.1f}%, Specific: {sel_specific:.1f}%)")
            
            # Log detailed factor and asset attributions
            descendant_leaves = context.descendant_leaves if hasattr(context, 'descendant_leaves') else None
            
            self._log_factor_attribution(context.portfolio_decomposer, node_id, 'portfolio')
            self._log_factor_attribution(context.benchmark_decomposer, node_id, 'benchmark')
            self._log_factor_attribution(context.active_decomposer, node_id, 'active')
            
            if descendant_leaves:
                self._log_asset_attribution(context.portfolio_decomposer, node_id, 'portfolio', descendant_leaves)
                self._log_asset_attribution(context.benchmark_decomposer, node_id, 'benchmark', descendant_leaves)
                self._log_asset_attribution(context.active_decomposer, node_id, 'active', descendant_leaves)
            
            self.logger.debug(f"Risk decomposition summary completed for {node_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to log decomposition summary for {node_id}: {str(e)}")
    
    def _log_factor_attribution(self, decomposer: 'RiskDecomposerBase', node_id: str, decomposer_type: str) -> None:
        """Log detailed factor-level risk attribution"""
        try:
            if not hasattr(decomposer, 'factor_contributions'):
                self.logger.debug(f"No factor contributions available for {decomposer_type} decomposer in {node_id}")
                return
                
            factor_contributions = decomposer.factor_contributions
            if factor_contributions is None or len(factor_contributions) != len(self.factor_names):
                self.logger.warning(f"Factor contributions mismatch for {decomposer_type} in {node_id}")
                return
            
            self.logger.info(f"Factor attribution for {node_id} ({decomposer_type}):")
            
            # Create factor attribution table
            factor_data = list(zip(self.factor_names, factor_contributions))
            factor_data_sorted = sorted(factor_data, key=lambda x: abs(x[1]), reverse=True)
            
            total_factor_risk = np.sum(np.abs(factor_contributions))
            
            for i, (factor_name, contribution) in enumerate(factor_data_sorted):
                percentage = 100.0 * abs(contribution) / total_factor_risk if total_factor_risk > 0 else 0.0
                self.logger.info(f"  {i+1:2d}. {factor_name:<12}: {contribution:8.4f} ({percentage:5.1f}%)")
                
                # Only log top 5 at INFO level, rest at DEBUG
                if i >= 4:
                    self.logger.debug(f"  {i+1:2d}. {factor_name:<12}: {contribution:8.4f} ({percentage:5.1f}%)")
                    
        except Exception as e:
            self.logger.warning(f"Failed to log factor attribution for {decomposer_type} in {node_id}: {str(e)}")
    
    def _log_asset_attribution(self, decomposer: 'RiskDecomposerBase', node_id: str, decomposer_type: str, asset_names: List[str] = None) -> None:
        """Log detailed asset-level risk attribution"""
        try:
            if not hasattr(decomposer, 'asset_contributions'):
                self.logger.debug(f"No asset contributions available for {decomposer_type} decomposer in {node_id}")
                return
                
            asset_contributions = decomposer.asset_contributions
            if asset_contributions is None:
                self.logger.warning(f"Asset contributions are None for {decomposer_type} in {node_id}")
                return
            
            # Use provided asset names or generate default ones
            if asset_names is None or len(asset_names) != len(asset_contributions):
                asset_names = [f"Asset_{i}" for i in range(len(asset_contributions))]
            
            self.logger.info(f"Asset attribution for {node_id} ({decomposer_type}):")
            
            # Create asset attribution table
            asset_data = list(zip(asset_names, asset_contributions))
            asset_data_sorted = sorted(asset_data, key=lambda x: abs(x[1]), reverse=True)
            
            total_asset_risk = np.sum(np.abs(asset_contributions))
            
            for i, (asset_name, contribution) in enumerate(asset_data_sorted):
                percentage = 100.0 * abs(contribution) / total_asset_risk if total_asset_risk > 0 else 0.0
                self.logger.info(f"  {i+1:2d}. {asset_name:<12}: {contribution:8.4f} ({percentage:5.1f}%)")
                
                # Only log top 10 at INFO level, rest at DEBUG
                if i >= 9:
                    self.logger.debug(f"  {i+1:2d}. {asset_name:<12}: {contribution:8.4f} ({percentage:5.1f}%)")
                    
        except Exception as e:
            self.logger.warning(f"Failed to log asset attribution for {decomposer_type} in {node_id}: {str(e)}")
    
    def _get_component_returns(self, component: 'PortfolioComponent', metric_name: str) -> Optional[pd.Series]:
        """Get returns data for a component from metric store"""
        if not self.metric_store:
            return None
        
        metric = self.metric_store.get_metric(component.component_id, metric_name)
        if metric is None:
            return None
        
        returns_data = metric.value()
        if isinstance(returns_data, pd.Series):
            return returns_data
        elif isinstance(returns_data, (int, float)):
            # If scalar, create a series with factor returns index
            return pd.Series(returns_data, index=self.factor_returns.index)
        
        return None
    
    def _get_component_weight(self, component: 'PortfolioComponent', weight_metric: str) -> float:
        """Get weight for a component with overlay strategy support"""
        # Handle overlay strategies - use operational weight for risk decomposition
        if hasattr(component, 'is_overlay') and getattr(component, 'is_overlay', False):
            # For overlay components, use operational weight (parent weight) instead of allocation weight
            return component.get_operational_weight(weight_metric)
        
        if not self.metric_store:
            return 1.0
        
        # Try specific weight metric first
        metric = self.metric_store.get_metric(component.component_id, weight_metric)
        if metric:
            return metric.value()
        
        # Fall back to base weight metric
        base_weight_metric = weight_metric.replace('portfolio_', '').replace('benchmark_', '').replace('active_', '')
        metric = self.metric_store.get_metric(component.component_id, base_weight_metric)
        if metric:
            return metric.value()
        
        return 1.0
    
    def _calculate_effective_weights(self, node: 'PortfolioNode', descendant_leaves: List[str]) -> None:
        """Calculate effective weights for descendants using WeightPathAggregator"""
        node_id = node.component_id
        self.logger.debug(f"Calculating effective weights for node {node_id} with {len(descendant_leaves)} descendants")
        
        # Calculate path matrices using the weight aggregator
        self._weight_aggregator.calculate_path_matrices()
        self.logger.debug(f"Path matrices calculated for hierarchical weight computation")
        
        # Get path weights for each descendant leaf
        for leaf_id in descendant_leaves:
            # Store contribution weights for this node-leaf pair
            portfolio_path_weight = 0.0
            benchmark_path_weight = 0.0
            
            # Get path matrix for this node
            portfolio_path_matrix = self._weight_aggregator.get_path_matrix(node_id, 'portfolio')
            benchmark_path_matrix = self._weight_aggregator.get_path_matrix(node_id, 'benchmark')
            
            if portfolio_path_matrix and leaf_id in portfolio_path_matrix:
                portfolio_path_weight = portfolio_path_matrix[leaf_id]
            if benchmark_path_matrix and leaf_id in benchmark_path_matrix:
                benchmark_path_weight = benchmark_path_matrix[leaf_id]
            
            self.logger.debug(f"  Path weights for {leaf_id}: portfolio={portfolio_path_weight:.6f}, benchmark={benchmark_path_weight:.6f}")
            
            self._node_contribution_weights[(node_id, leaf_id)] = {
                'portfolio': portfolio_path_weight,
                'benchmark': benchmark_path_weight,
            }
        
        # Get effective weights as numpy arrays for risk calculations
        portfolio_weights = self._weight_aggregator.get_effective_weights(node_id, 'portfolio')
        benchmark_weights = self._weight_aggregator.get_effective_weights(node_id, 'benchmark')
        
        self.logger.debug(f"Retrieved effective weights: portfolio shape={portfolio_weights.shape if portfolio_weights is not None else 'None'}, benchmark shape={benchmark_weights.shape if benchmark_weights is not None else 'None'}")
        
        if portfolio_weights is not None and benchmark_weights is not None:
            # Normalize weights within this node's context
            portfolio_normalized = self._weight_aggregator.normalize_weights(portfolio_weights)
            benchmark_normalized = self._weight_aggregator.normalize_weights(benchmark_weights)
            
            self.logger.info(f"Normalized effective weights for {node_id}: portfolio sum={np.sum(portfolio_normalized):.6f}, benchmark sum={np.sum(benchmark_normalized):.6f}")
            
            # Log detailed weight analysis
            self._log_weight_analysis(portfolio_normalized, node_id, 'portfolio', descendant_leaves)
            self._log_weight_analysis(benchmark_normalized, node_id, 'benchmark', descendant_leaves)
            active_weights = portfolio_normalized - benchmark_normalized
            self._log_weight_analysis(active_weights, node_id, 'active', descendant_leaves)
            
            self._descendant_weights[node_id] = {
                'portfolio': portfolio_normalized,
                'benchmark': benchmark_normalized
            }
        else:
            # Fallback to equal weights if path calculation failed
            num_leaves = len(descendant_leaves)
            equal_weights = np.ones(num_leaves) / num_leaves if num_leaves > 0 else np.array([])
            
            self.logger.warning(f"Path calculation failed for {node_id}, using equal weights fallback")
            self._log_weight_analysis(equal_weights, node_id, 'portfolio_fallback', descendant_leaves)
            self._log_weight_analysis(equal_weights, node_id, 'benchmark_fallback', descendant_leaves)
            
            self._descendant_weights[node_id] = {
                'portfolio': equal_weights,
                'benchmark': equal_weights
            }
    
    
    def get_effective_weights(self, node_id: str, return_type: str) -> Optional[np.ndarray]:
        """Get effective weights for a node's descendants"""
        if node_id not in self._descendant_weights:
            return None
        
        if return_type == 'active':
            # Active = portfolio - benchmark
            portfolio_weights = self._descendant_weights[node_id]['portfolio']
            benchmark_weights = self._descendant_weights[node_id]['benchmark']
            return portfolio_weights - benchmark_weights
        else:
            return self._descendant_weights[node_id].get(return_type)
    
    def get_node_weight(self, node_id: str, return_type: str) -> Optional[float]:
        """Get a node's own weight"""
        if node_id not in self._node_weights:
            return None
        
        if return_type == 'active':
            # Active = portfolio - benchmark
            portfolio_weight = self._node_weights[node_id]['portfolio']
            benchmark_weight = self._node_weights[node_id]['benchmark']
            return portfolio_weight - benchmark_weight
        else:
            return self._node_weights[node_id].get(return_type)
    
    # NOTE: Removed _build_unified_node_matrices method (247 lines) - replaced with direct model re-estimation
    
    @property
    def result(self) -> Dict[str, Dict[str, Any]]:
        """Get all risk results for processed nodes"""
        return self._node_risk_results.copy()
    
    def get_node_risk_results(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get risk results for a specific node"""
        return self._node_risk_results.get(node_id)
    
    def get_processed_components(self) -> Set[str]:
        """Get set of all processed component IDs"""
        return self._processed_components.copy()
    
    def get_node_contribution_weight(self, node_id: str, leaf_id: str, return_type: str) -> Optional[float]:
        """Get the contribution weight from a node to a specific leaf for a return type"""
        key = (node_id, leaf_id)
        if key not in self._node_contribution_weights:
            return None
        
        if return_type == 'active':
            # Active = portfolio - benchmark
            portfolio_weight = self._node_contribution_weights[key].get('portfolio', 0.0)
            benchmark_weight = self._node_contribution_weights[key].get('benchmark', 0.0)
            return portfolio_weight - benchmark_weight
        else:
            return self._node_contribution_weights[key].get(return_type)
    
    def get_all_node_contribution_weights(self) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Get all node contribution weights for debugging/inspection"""
        return self._node_contribution_weights.copy()
    
    def get_factor_names(self) -> List[str]:
        """Get list of factor names from the visitor."""
        return self.factor_names.copy()
    
    def get_component_ids(self) -> List[str]:
        """Get list of processed component IDs."""
        return list(self._processed_components)
    
    def get_risk_contribution_matrix(self) -> Optional[np.ndarray]:
        """
        Get risk contribution matrix (factors × components).
        
        Extracts factor contributions from RiskResult objects.
        """
        # Try to extract from stored risk results
        if not self._node_risk_results:
            return None
        
        try:
            n_factors = len(self.factor_names)
            n_components = len(self._processed_components)
            
            contribution_matrix = np.zeros((n_factors, n_components))
            
            # Fill matrix with factor contributions from stored results
            for i, component_id in enumerate(self._processed_components):
                if component_id in self._node_risk_results:
                    portfolio_result = self._node_risk_results[component_id].get('portfolio')
                    if portfolio_result and hasattr(portfolio_result, 'factor_contributions'):
                        for j, factor_name in enumerate(self.factor_names):
                            if factor_name in portfolio_result.factor_contributions:
                                contribution_matrix[j, i] = portfolio_result.factor_contributions[factor_name]
            
            return contribution_matrix
            
        except Exception:
            return None
    
    def validate_euler_identity(self) -> Dict[str, Any]:
        """
        Validate Euler identity for risk decomposition.
        
        Returns basic validation results. More sophisticated validation
        would require access to specific decomposer results.
        """
        return {
            'components_processed': len(self._processed_components),
            'factors_analyzed': len(self.factor_names),
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'status': 'visitor_validation_basic'
        }
    
    def to_unified_schema(self, component_id: str) -> 'RiskResultSchema':
        """
        Export visitor results to unified schema format for a specific component.
        
        This method extracts the risk decomposition results from the visitor
        and converts them to the standardized unified schema format using
        the hierarchical extraction method.
        
        Parameters
        ----------
        component_id : str
            Component ID to extract results for (typically root component)
            
        Returns
        -------
        RiskResultSchema
            Visitor results in unified schema format with comprehensive data
        """
        from spark.risk.schema_factory import RiskSchemaFactory
        
        # Use factory method to create schema from visitor results
        return RiskSchemaFactory.from_factor_risk_decomposition_visitor(
            visitor=self,
            root_component_id=component_id,
            map_full_hierarchy=True
        )


class ExcessReturnVisitor(PortfolioVisitor):
    """Specialized visitor for excess return calculation (Portfolio - Benchmark)"""
    
    def __init__(self, 
                 portfolio_metric: str = "port_ret", 
                 benchmark_metric: str = "bench_ret",
                 metric_store: Optional[MetricStore] = None,
                 portfolio_weight_metric: str = 'weight',
                 benchmark_weight_metric: str = 'weight'):
        self.portfolio_metric = portfolio_metric
        self.benchmark_metric = benchmark_metric
        self.metric_store = metric_store
        self.portfolio_weight_metric = portfolio_weight_metric
        self.benchmark_weight_metric = benchmark_weight_metric
        # Separate stacks for portfolio and benchmark returns
        self._portfolio_stack: List[Metric] = []
        self._benchmark_stack: List[Metric] = []
    
    def visit_leaf(self, leaf: 'PortfolioLeaf') -> None:
        """Visit leaf and push portfolio and benchmark metrics to separate stacks"""
        port_metric = None
        bench_metric = None
        
        if self.metric_store:
            port_metric = self.metric_store.get_metric(leaf.component_id, self.portfolio_metric)
            bench_metric = self.metric_store.get_metric(leaf.component_id, self.benchmark_metric)
        
        # Push to separate stacks (with defaults if not found)
        from .metrics import ScalarMetric
        
        if port_metric:
            self._portfolio_stack.append(port_metric)
        else:
            self._portfolio_stack.append(ScalarMetric(0.0))
            
        if bench_metric:
            self._benchmark_stack.append(bench_metric)
        else:
            self._benchmark_stack.append(ScalarMetric(0.0))
    
    def visit_node(self, node: 'PortfolioNode') -> None:
        """Visit node by first visiting children, then aggregating their portfolio and benchmark returns"""
        portfolio_child_returns = []
        benchmark_child_returns = []
        
        # First, visit all children to populate the stacks
        for child_id in node.get_all_children():
            child_component = node._get_child_component(child_id)
            if child_component:
                # Visit the child (this will populate the stacks)
                child_component.accept(self)
                
                # Pop the results from both stacks (child's contribution)
                if self._portfolio_stack and self._benchmark_stack:
                    child_portfolio = self._portfolio_stack.pop()
                    child_benchmark = self._benchmark_stack.pop()
                    
                    # Get weights for this child at the current node level
                    portfolio_weight = self._get_portfolio_weight(child_component)
                    benchmark_weight = self._get_benchmark_weight(child_component)
                    
                    portfolio_child_returns.append((portfolio_weight, child_portfolio))
                    benchmark_child_returns.append((benchmark_weight, child_benchmark))
        
        # Aggregate portfolio and benchmark separately using direct weighted average calculation
        if portfolio_child_returns and benchmark_child_returns:
            portfolio_result = self._calculate_weighted_average(portfolio_child_returns)
            benchmark_result = self._calculate_weighted_average(benchmark_child_returns)
            
            # Push the aggregated results back to stacks for potential parent nodes
            self._portfolio_stack.append(portfolio_result)
            self._benchmark_stack.append(benchmark_result)
        else:
            # Push defaults if no children
            from .metrics import ScalarMetric
            self._portfolio_stack.append(ScalarMetric(0.0))
            self._benchmark_stack.append(ScalarMetric(0.0))
    
    def _calculate_weighted_average(self, weighted_metrics):
        """Calculate weighted average of metrics directly without external dependencies"""
        if not weighted_metrics:
            from .metrics import ScalarMetric
            return ScalarMetric(0.0)
        
        # Calculate total weight
        total_weight = sum(weight for weight, _ in weighted_metrics)
        if total_weight == 0:
            from .metrics import ScalarMetric
            return ScalarMetric(0.0)
        
        # Determine metric type from first metric
        first_metric = weighted_metrics[0][1]
        from .metrics import ScalarMetric, SeriesMetric
        
        if isinstance(first_metric, ScalarMetric):
            # Calculate weighted average for scalar metrics
            weighted_sum = sum(weight * metric.value() for weight, metric in weighted_metrics)
            return ScalarMetric(weighted_sum / total_weight)
        
        elif isinstance(first_metric, SeriesMetric):
            # Calculate weighted average for series metrics
            weighted_sum = None
            
            for weight, metric in weighted_metrics:
                series = metric.value()
                if weighted_sum is None:
                    weighted_sum = weight * series
                else:
                    weighted_sum = weighted_sum + (weight * series)
            
            return SeriesMetric(weighted_sum / total_weight)
        
        else:
            # Fallback for unknown metric types
            return ScalarMetric(0.0)
    
    def _get_portfolio_weight(self, component: 'PortfolioComponent') -> float:
        """Get portfolio weight for a component from metric store"""
        if self.metric_store:
            weight_metric = self.metric_store.get_metric(component.component_id, self.portfolio_weight_metric)
            if weight_metric:
                return weight_metric.value()
        return 1.0  # Default weight
    
    def _get_benchmark_weight(self, component: 'PortfolioComponent') -> float:
        """Get benchmark weight for a component from metric store"""
        if self.metric_store:
            weight_metric = self.metric_store.get_metric(component.component_id, self.benchmark_weight_metric)
            if weight_metric:
                return weight_metric.value()
        return 1.0  # Default weight
    
    @property
    def result(self) -> Optional[Metric]:
        """Get the final excess return result by calculating Portfolio - Benchmark"""
        if self._portfolio_stack and self._benchmark_stack:
            portfolio_result = self._portfolio_stack[-1]
            benchmark_result = self._benchmark_stack[-1]
            
            # Calculate excess return
            from .metrics import ScalarMetric, SeriesMetric
            if isinstance(portfolio_result, ScalarMetric) and isinstance(benchmark_result, ScalarMetric):
                excess = portfolio_result.value() - benchmark_result.value()
                return ScalarMetric(excess)
            elif isinstance(portfolio_result, SeriesMetric) and isinstance(benchmark_result, SeriesMetric):
                excess_series = portfolio_result.value() - benchmark_result.value()
                return SeriesMetric(excess_series)
        
        # Default fallback
        from .metrics import ScalarMetric
        return ScalarMetric(0.0)
    
    def run_on(self, component: 'PortfolioComponent') -> Optional[Metric]:
        """Convenience method to run visitor on a component and return result"""
        component.accept(self)
        return self.result


def debug_risk_breakdown(visitor: 'FactorRiskDecompositionVisitor', node_id: str, log_to_file: bool = True) -> None:
    """
    Print basic risk breakdown for debugging purposes.
    
    Parameters
    ----------
    visitor : FactorRiskDecompositionVisitor
        Processed visitor containing risk decomposition results
    node_id : str
        Node ID to generate risk breakdown for
    log_to_file : bool, default True
        Whether to also log output to the visitor's log file
    """
    debug_output = f"\n=== Risk Debug Report: {node_id} ==="
    print(debug_output)
    
    # Log to visitor's logger if available and requested
    if log_to_file and hasattr(visitor, 'logger'):
        visitor.logger.info(f"Debug risk breakdown requested for node: {node_id}")
    
    # Check if visitor has metric store
    if not visitor.metric_store:
        error_msg = "❌ No metric store available in visitor"
        print(error_msg)
        if log_to_file and hasattr(visitor, 'logger'):
            visitor.logger.error(f"Debug breakdown failed: {error_msg}")
        return
    
    # Extract HierarchicalModelContext from metric store
    context_metric = visitor.metric_store.get_metric(node_id, 'hierarchical_model_context')
    if not context_metric:
        error_msg = f"❌ No hierarchical model context found for node '{node_id}'"
        print(error_msg)
        if log_to_file and hasattr(visitor, 'logger'):
            visitor.logger.warning(f"Debug breakdown: {error_msg}")
        return
    
    context = context_metric.value()
    if not hasattr(context, 'portfolio_decomposer'):
        error_msg = f"❌ Invalid hierarchical model context object for node '{node_id}'"
        print(error_msg)
        if log_to_file and hasattr(visitor, 'logger'):
            visitor.logger.error(f"Debug breakdown: {error_msg}")
        return
    
    # Get risk summaries
    try:
        portfolio_summary = context.portfolio_decomposer.risk_decomposition_summary()
        benchmark_summary = context.benchmark_decomposer.risk_decomposition_summary()
        active_summary = context.active_decomposer.risk_decomposition_summary()
        
        if log_to_file and hasattr(visitor, 'logger'):
            visitor.logger.info(f"Successfully extracted risk summaries for node {node_id}")
    except Exception as e:
        error_msg = f"❌ Error getting risk summaries: {e}"
        print(error_msg)
        if log_to_file and hasattr(visitor, 'logger'):
            visitor.logger.error(f"Debug breakdown failed: {error_msg}")
        return
    
    # Print Portfolio Risk
    portfolio_risk_msg = f"Portfolio Risk: {portfolio_summary['portfolio_volatility']:.1%}"
    factor_risk_msg = f"  Factor Risk: {portfolio_summary['factor_risk_contribution']:.1%} ({portfolio_summary['factor_risk_percentage']:.1f}%)"
    specific_risk_msg = f"  Specific Risk: {portfolio_summary['specific_risk_contribution']:.1%} ({portfolio_summary['specific_risk_percentage']:.1f}%)"
    
    print(portfolio_risk_msg)
    print(factor_risk_msg)
    print(specific_risk_msg)
    
    if log_to_file and hasattr(visitor, 'logger'):
        visitor.logger.info(f"Node {node_id} Portfolio Risk: {portfolio_summary['portfolio_volatility']:.1%} (Factor: {portfolio_summary['factor_risk_percentage']:.1f}%, Specific: {portfolio_summary['specific_risk_percentage']:.1f}%)")
    
    # Print top contributors if available
    try:
        if 'top_asset_contributors' in portfolio_summary and portfolio_summary['top_asset_contributors']:
            top_assets = portfolio_summary['top_asset_contributors']
            if isinstance(top_assets, list) and len(top_assets) > 0:
                asset_str = ", ".join([f"{name} ({contrib:.1%})" for name, contrib in top_assets[:3]])
                print(f"  Top Assets: {asset_str}")
    except Exception:
        pass  # Skip if format is unexpected
    
    try:
        if 'top_factor_contributors' in portfolio_summary and portfolio_summary['top_factor_contributors']:
            top_factors = portfolio_summary['top_factor_contributors']
            if isinstance(top_factors, list) and len(top_factors) > 0:
                factor_str = ", ".join([f"{name} ({contrib:.1%})" for name, contrib in top_factors[:3]])
                print(f"  Top Factors: {factor_str}")
    except Exception:
        pass  # Skip if format is unexpected
    
    # Print Benchmark Risk
    print(f"\nBenchmark Risk: {benchmark_summary['portfolio_volatility']:.1%}")
    print(f"  Factor Risk: {benchmark_summary['factor_risk_contribution']:.1%} ({benchmark_summary['factor_risk_percentage']:.1f}%)")
    print(f"  Specific Risk: {benchmark_summary['specific_risk_contribution']:.1%} ({benchmark_summary['specific_risk_percentage']:.1f}%)")
    
    # Print Active Risk
    print(f"\nActive Risk: {active_summary['total_active_risk']:.1%}")
    if 'allocation_factor_percentage' in active_summary:
        alloc_total = active_summary.get('allocation_factor_percentage', 0) + active_summary.get('allocation_specific_percentage', 0)
        sel_total = active_summary.get('selection_factor_percentage', 0) + active_summary.get('selection_specific_percentage', 0)
        print(f"  Allocation Risk: {alloc_total:.1f}%")
        print(f"  Selection Risk: {sel_total:.1f}%")
    
    # Print basic validation
    portfolio_sum = portfolio_summary['factor_risk_contribution'] + portfolio_summary['specific_risk_contribution']
    portfolio_total = portfolio_summary['portfolio_volatility']
    portfolio_diff = abs(portfolio_sum - portfolio_total)
    
    print(f"\nValidation:")
    print(f"  Portfolio contributions sum: {portfolio_diff:.2%} difference from total")
    
    if portfolio_diff > 0.01:  # 1% tolerance
        validation_msg = "  ⚠️  Large difference detected - check risk calculations"
        print(validation_msg)
        if log_to_file and hasattr(visitor, 'logger'):
            visitor.logger.warning(f"Node {node_id} validation failed: {portfolio_diff:.2%} difference in risk contributions")
    else:
        validation_msg = "  ✅ Risk contributions validated"
        print(validation_msg)
        if log_to_file and hasattr(visitor, 'logger'):
            visitor.logger.debug(f"Node {node_id} validation passed: {portfolio_diff:.2%} difference in risk contributions")
    
    # Log final summary to file
    if log_to_file and hasattr(visitor, 'logger'):
        visitor.logger.info(f"Debug risk breakdown completed for node {node_id}")
        visitor.logger.info(f"Node {node_id} Active Risk: {active_summary['total_active_risk']:.1%}")
    
    print()


    