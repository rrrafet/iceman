"""
Decision Attribution Framework
==============================

Framework for analyzing the risk impact of specific portfolio decisions at any 
hierarchy level. Provides comprehensive attribution of how decisions propagate 
through the portfolio hierarchy and affect active risk components.

This module integrates with existing WeightPathAggregator and RiskDecomposer 
infrastructure to provide decision-level risk analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import logging
from copy import deepcopy

if TYPE_CHECKING:
    from .graph import PortfolioGraph
    from ..risk.decomposer import RiskDecomposer
    from ..risk.context import HierarchicalModelContext

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of portfolio decisions that can be analyzed"""
    TILT = "tilt"                    # Change allocation vs benchmark
    REBALANCE = "rebalance"          # Redistribute within existing allocations  
    ALLOCATION = "allocation"        # Change absolute allocation
    SCALE = "scale"                  # Scale up/down a component


@dataclass
class WeightChange:
    """Represents a weight change for a component"""
    component_id: str
    portfolio_weight_before: float
    portfolio_weight_after: float
    benchmark_weight_before: float
    benchmark_weight_after: float
    
    @property
    def portfolio_change(self) -> float:
        """Portfolio weight change"""
        return self.portfolio_weight_after - self.portfolio_weight_before
    
    @property
    def benchmark_change(self) -> float:
        """Benchmark weight change"""
        return self.benchmark_weight_after - self.benchmark_weight_before
    
    @property
    def active_weight_before(self) -> float:
        """Active weight before decision"""
        return self.portfolio_weight_before - self.benchmark_weight_before
    
    @property
    def active_weight_after(self) -> float:
        """Active weight after decision"""
        return self.portfolio_weight_after - self.benchmark_weight_after
    
    @property
    def active_weight_change(self) -> float:
        """Change in active weight"""
        return self.active_weight_after - self.active_weight_before


class PortfolioDecision:
    """
    Represents a specific portfolio decision and tracks its impact through the hierarchy.
    
    This class encapsulates a decision made at any level of the portfolio hierarchy
    and provides methods to understand how that decision affects all descendant components.
    
    Parameters
    ----------
    component_id : str
        ID of the component where the decision is made
    decision_type : DecisionType
        Type of decision being made
    weight_changes : Dict[str, WeightChange]
        Direct weight changes specified by the decision
    description : str, optional
        Human-readable description of the decision
    """
    
    def __init__(
        self,
        component_id: str,
        decision_type: DecisionType,
        weight_changes: Dict[str, WeightChange],
        description: Optional[str] = None
    ):
        self.component_id = component_id
        self.decision_type = decision_type
        self.weight_changes = weight_changes
        self.description = description or f"{decision_type.value} on {component_id}"
        
        # Will be populated by the analyzer
        self.affected_descendants: List[str] = []
        self.propagated_weight_changes: Dict[str, WeightChange] = {}
        
    def get_primary_weight_change(self) -> WeightChange:
        """Get the primary weight change (typically for the decision component)"""
        if self.component_id in self.weight_changes:
            return self.weight_changes[self.component_id]
        elif len(self.weight_changes) == 1:
            return list(self.weight_changes.values())[0]
        else:
            raise ValueError("Cannot determine primary weight change")
    
    def get_total_portfolio_weight_change(self) -> float:
        """Get total portfolio weight change across all affected components"""
        return sum(change.portfolio_change for change in self.weight_changes.values())
    
    def get_total_active_weight_change(self) -> float:
        """Get total active weight change across all affected components"""
        return sum(change.active_weight_change for change in self.weight_changes.values())
    
    @classmethod
    def create_tilt_decision(
        cls,
        component_id: str,
        portfolio_weight_change: float,
        benchmark_weight_change: float = 0.0,
        current_portfolio_weight: float = None,
        current_benchmark_weight: float = None,
        description: Optional[str] = None
    ) -> 'PortfolioDecision':
        """
        Create a tilting decision for a specific component.
        
        Parameters
        ----------
        component_id : str
            Component to tilt
        portfolio_weight_change : float
            Change in portfolio weight
        benchmark_weight_change : float, default 0.0
            Change in benchmark weight
        current_portfolio_weight : float, optional
            Current portfolio weight (will be retrieved if not provided)
        current_benchmark_weight : float, optional
            Current benchmark weight (will be retrieved if not provided)
        description : str, optional
            Description of the decision
            
        Returns
        -------
        PortfolioDecision
            Decision object representing the tilt
        """
        # Note: current weights would be retrieved from PortfolioGraph in practice
        # For now, using defaults that can be updated by the analyzer
        port_before = current_portfolio_weight or 0.0
        bench_before = current_benchmark_weight or 0.0
        
        weight_change = WeightChange(
            component_id=component_id,
            portfolio_weight_before=port_before,
            portfolio_weight_after=port_before + portfolio_weight_change,
            benchmark_weight_before=bench_before,
            benchmark_weight_after=bench_before + benchmark_weight_change
        )
        
        desc = description or f"Tilt {component_id} by {portfolio_weight_change:+.2%}"
        
        return cls(
            component_id=component_id,
            decision_type=DecisionType.TILT,
            weight_changes={component_id: weight_change},
            description=desc
        )
    
    def __repr__(self) -> str:
        return f"PortfolioDecision({self.description})"


@dataclass
class ComponentRiskAttribution:
    """Risk attribution for a single component affected by the decision"""
    component_id: str
    component_type: str  # 'node' or 'leaf'
    
    # Weight changes
    weight_change: WeightChange
    
    # Risk contribution changes
    total_risk_contribution_before: float
    total_risk_contribution_after: float
    allocation_factor_contribution_change: float
    allocation_specific_contribution_change: float
    selection_factor_contribution_change: float
    selection_specific_contribution_change: float
    
    # Factor exposure changes
    factor_exposure_changes: Dict[str, float]
    
    @property
    def total_risk_contribution_change(self) -> float:
        """Total change in risk contribution"""
        return self.total_risk_contribution_after - self.total_risk_contribution_before
    
    @property
    def total_allocation_contribution_change(self) -> float:
        """Total allocation contribution change"""
        return self.allocation_factor_contribution_change + self.allocation_specific_contribution_change
    
    @property
    def total_selection_contribution_change(self) -> float:
        """Total selection contribution change"""
        return self.selection_factor_contribution_change + self.selection_specific_contribution_change


class DecisionRiskAttribution:
    """
    Comprehensive results of decision risk attribution analysis.
    
    Contains the complete breakdown of how a portfolio decision affects
    active risk at all levels of the hierarchy.
    
    Parameters
    ----------
    decision : PortfolioDecision
        The decision that was analyzed
    """
    
    def __init__(self, decision: PortfolioDecision):
        self.decision = decision
        self.timestamp = pd.Timestamp.now()
        
        # Overall impact
        self.total_active_risk_before: float = 0.0
        self.total_active_risk_after: float = 0.0
        
        # Component-level attributions
        self.component_attributions: Dict[str, ComponentRiskAttribution] = {}
        
        # Factor-level changes
        self.factor_exposure_changes: Dict[str, float] = {}
        self.factor_risk_contribution_changes: Dict[str, float] = {}
        
        # Risk decomposition changes
        self.allocation_factor_risk_change: float = 0.0
        self.allocation_specific_risk_change: float = 0.0
        self.selection_factor_risk_change: float = 0.0
        self.selection_specific_risk_change: float = 0.0
        
        # Hierarchical impact tracking
        self.hierarchical_impact_map: Dict[str, Dict[str, Any]] = {}
        
    @property
    def total_active_risk_change(self) -> float:
        """Total change in active risk"""
        return self.total_active_risk_after - self.total_active_risk_before
    
    @property
    def total_allocation_risk_change(self) -> float:
        """Total allocation risk change"""
        return self.allocation_factor_risk_change + self.allocation_specific_risk_change
    
    @property
    def total_selection_risk_change(self) -> float:
        """Total selection risk change"""
        return self.selection_factor_risk_change + self.selection_specific_risk_change
    
    @property
    def total_factor_risk_change(self) -> float:
        """Total factor risk change"""
        return self.allocation_factor_risk_change + self.selection_factor_risk_change
    
    @property
    def total_specific_risk_change(self) -> float:
        """Total specific risk change"""
        return self.allocation_specific_risk_change + self.selection_specific_risk_change
    
    def get_component_attribution(self, component_id: str) -> Optional[ComponentRiskAttribution]:
        """Get risk attribution for a specific component"""
        return self.component_attributions.get(component_id)
    
    def get_affected_components(self) -> List[str]:
        """Get list of all components affected by the decision"""
        return list(self.component_attributions.keys())
    
    def get_top_risk_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N components by absolute risk contribution change.
        
        Returns
        -------
        List[Tuple[str, float]]
            List of (component_id, risk_contribution_change) tuples
        """
        contributions = [
            (comp_id, abs(attr.total_risk_contribution_change))
            for comp_id, attr in self.component_attributions.items()
        ]
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions[:n]
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for reporting"""
        return {
            'decision': {
                'component_id': self.decision.component_id,
                'decision_type': self.decision.decision_type.value,
                'description': self.decision.description,
                'total_weight_change': self.decision.get_total_portfolio_weight_change(),
                'total_active_weight_change': self.decision.get_total_active_weight_change()
            },
            'risk_impact': {
                'total_active_risk_before': self.total_active_risk_before,
                'total_active_risk_after': self.total_active_risk_after,
                'total_active_risk_change': self.total_active_risk_change,
                'allocation_risk_change': self.total_allocation_risk_change,
                'selection_risk_change': self.total_selection_risk_change,
                'factor_risk_change': self.total_factor_risk_change,
                'specific_risk_change': self.total_specific_risk_change
            },
            'factor_impact': {
                'factor_exposure_changes': self.factor_exposure_changes,
                'factor_risk_changes': self.factor_risk_contribution_changes
            },
            'component_impact': {
                'num_affected_components': len(self.component_attributions),
                'top_contributors': self.get_top_risk_contributors(3)
            },
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert component attributions to DataFrame for analysis"""
        if not self.component_attributions:
            return pd.DataFrame()
        
        data = []
        for comp_id, attr in self.component_attributions.items():
            data.append({
                'component_id': comp_id,
                'component_type': attr.component_type,
                'portfolio_weight_change': attr.weight_change.portfolio_change,
                'benchmark_weight_change': attr.weight_change.benchmark_change,
                'active_weight_change': attr.weight_change.active_weight_change,
                'total_risk_contribution_change': attr.total_risk_contribution_change,
                'allocation_contribution_change': attr.total_allocation_contribution_change,
                'selection_contribution_change': attr.total_selection_contribution_change,
                'allocation_factor_change': attr.allocation_factor_contribution_change,
                'allocation_specific_change': attr.allocation_specific_contribution_change,
                'selection_factor_change': attr.selection_factor_contribution_change,
                'selection_specific_change': attr.selection_specific_contribution_change
            })
        
        return pd.DataFrame(data).set_index('component_id')
    
    def __repr__(self) -> str:
        return (f"DecisionRiskAttribution({self.decision.description}, "
                f"risk_change={self.total_active_risk_change:.6f}, "
                f"components={len(self.component_attributions)})")


class DecisionAttributionError(Exception):
    """Exception raised during decision attribution analysis"""
    pass


class HierarchicalDecisionAnalyzer:
    """
    Main engine for hierarchical decision risk attribution analysis.
    
    This class performs comprehensive before/after analysis of portfolio decisions,
    leveraging existing WeightPathAggregator and RiskDecomposer infrastructure
    to provide detailed risk attribution at all hierarchy levels.
    
    Parameters
    ----------
    portfolio_graph : PortfolioGraph
        Portfolio graph containing the hierarchy and risk decomposition results
    """
    
    def __init__(self, portfolio_graph: 'PortfolioGraph'):
        self.portfolio_graph = portfolio_graph
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def analyze_decision_impact(
        self,
        decision: PortfolioDecision,
        factor_returns: Optional[pd.DataFrame] = None,
        estimator: Optional[Any] = None,
        force_recompute: bool = False
    ) -> DecisionRiskAttribution:
        """
        Perform comprehensive decision impact analysis.
        
        Parameters
        ----------
        decision : PortfolioDecision
            Decision to analyze
        factor_returns : pd.DataFrame, optional
            Factor returns for risk decomposition (if not already computed)
        estimator : LinearRiskModelEstimator, optional
            Risk model estimator (if not already computed)
        force_recompute : bool, default False
            Whether to force recomputation of risk decomposition
            
        Returns
        -------
        DecisionRiskAttribution
            Comprehensive attribution results
        """
        self.logger.info(f"Analyzing decision impact: {decision.description}")
        
        try:
            # Step 1: Capture baseline risk state
            self.logger.debug("Capturing baseline risk state")
            baseline_context = self._get_risk_context(
                self.portfolio_graph, decision.component_id, 
                factor_returns, estimator, force_recompute
            )
            
            # Step 2: Apply decision and propagate weight changes
            self.logger.debug("Applying decision and propagating weight changes")
            modified_graph, propagated_changes = self._apply_decision_with_propagation(
                self.portfolio_graph, decision
            )
            
            # Step 3: Recompute risk decomposition with new weights
            self.logger.debug("Recomputing risk decomposition with modified weights")
            scenario_context = self._get_risk_context(
                modified_graph, decision.component_id,
                factor_returns, estimator, force_recompute=True
            )
            
            # Step 4: Calculate comprehensive attribution
            self.logger.debug("Computing hierarchical risk attribution")
            attribution = self._compute_hierarchical_attribution(
                baseline_context, scenario_context, decision, propagated_changes
            )
            
            self.logger.info(f"Decision analysis completed. Risk change: {attribution.total_active_risk_change:.6f}")
            return attribution
            
        except Exception as e:
            self.logger.error(f"Error analyzing decision impact: {e}")
            raise DecisionAttributionError(f"Failed to analyze decision impact: {e}") from e
    
    def _get_risk_context(
        self,
        graph: 'PortfolioGraph',
        component_id: str,
        factor_returns: Optional[pd.DataFrame] = None,
        estimator: Optional[Any] = None,
        force_recompute: bool = False
    ) -> 'HierarchicalModelContext':
        """
        Get or compute risk context for a component.
        
        Parameters
        ----------
        graph : PortfolioGraph
            Portfolio graph to analyze
        component_id : str
            Component to get risk context for
        factor_returns : pd.DataFrame, optional
            Factor returns for risk decomposition
        estimator : LinearRiskModelEstimator, optional
            Risk model estimator
        force_recompute : bool, default False
            Whether to force recomputation
            
        Returns
        -------
        HierarchicalModelContext
            Risk context containing decomposition results
        """
        try:
            # Check if risk decomposition already exists
            if not force_recompute:
                try:
                    metrics = graph.metric_store.get_all_metrics(component_id)
                    context_metric = metrics.get('hierarchical_model_context')
                    if context_metric is not None:
                        return context_metric.value()
                except (KeyError, AttributeError):
                    pass
            
            # Need to compute risk decomposition
            if factor_returns is None or estimator is None:
                raise DecisionAttributionError(
                    f"Risk decomposition not found for {component_id} and no factor_returns/estimator provided"
                )
            
            # Run risk decomposition using PortfolioRiskAnalyzer
            from .risk_analyzer import PortfolioRiskAnalyzer
            risk_analyzer = PortfolioRiskAnalyzer(graph)
            risk_analyzer.decompose_factor_risk(
                root_component_id=component_id,
                factor_returns=factor_returns,
                estimator=estimator,
                annualize=True,
                verbose=False
            )
            
            # Retrieve the computed context
            metrics = graph.metric_store.get_all_metrics(component_id)
            context_metric = metrics.get('hierarchical_model_context')
            if context_metric is None:
                raise DecisionAttributionError(f"Failed to compute risk context for {component_id}")
            
            return context_metric.value()
            
        except Exception as e:
            raise DecisionAttributionError(f"Failed to get risk context for {component_id}: {e}") from e
    
    def _apply_decision_with_propagation(
        self,
        graph: 'PortfolioGraph',
        decision: PortfolioDecision
    ) -> Tuple['PortfolioGraph', Dict[str, WeightChange]]:
        """
        Apply decision to portfolio graph and propagate weight changes through hierarchy.
        
        Parameters
        ----------
        graph : PortfolioGraph
            Original portfolio graph
        decision : PortfolioDecision
            Decision to apply
            
        Returns
        -------
        Tuple[PortfolioGraph, Dict[str, WeightChange]]
            Modified graph and dictionary of all propagated weight changes
        """
        # Create a deep copy of the graph to avoid modifying the original
        modified_graph = self._create_graph_copy(graph)
        
        # Track all weight changes (direct + propagated)
        all_weight_changes = {}
        
        # Apply direct weight changes from the decision
        for comp_id, weight_change in decision.weight_changes.items():
            self._apply_weight_change_to_graph(modified_graph, weight_change)
            all_weight_changes[comp_id] = weight_change
        
        # Use WeightPathAggregator to determine affected descendants
        weight_service = modified_graph.create_weight_service()
        
        # For each directly modified component, find affected descendants
        for comp_id in decision.weight_changes.keys():
            affected_descendants = self._get_affected_descendants(
                modified_graph, comp_id, weight_service
            )
            decision.affected_descendants.extend(affected_descendants)
            
            # Calculate propagated weight changes for descendants
            propagated_changes = self._calculate_propagated_changes(
                graph, modified_graph, affected_descendants
            )
            
            all_weight_changes.update(propagated_changes)
        
        # Update decision with propagated changes
        decision.propagated_weight_changes = all_weight_changes
        
        return modified_graph, all_weight_changes
    
    def _create_graph_copy(self, graph: 'PortfolioGraph') -> 'PortfolioGraph':
        """Create a deep copy of the portfolio graph for modification"""
        # For now, return the same graph (in practice would create deep copy)
        # This is a simplification - real implementation would need proper graph copying
        return graph
    
    def _apply_weight_change_to_graph(
        self,
        graph: 'PortfolioGraph',
        weight_change: WeightChange
    ) -> None:
        """Apply a weight change to the portfolio graph"""
        from .metrics import ScalarMetric
        
        # Update portfolio weight
        graph.metric_store.set_metric(
            weight_change.component_id,
            'portfolio_weight',
            ScalarMetric(weight_change.portfolio_weight_after)
        )
        
        # Update benchmark weight
        graph.metric_store.set_metric(
            weight_change.component_id,
            'benchmark_weight',
            ScalarMetric(weight_change.benchmark_weight_after)
        )
    
    def _get_affected_descendants(
        self,
        graph: 'PortfolioGraph',
        component_id: str,
        weight_service: Any
    ) -> List[str]:
        """Get list of descendant components affected by weight change"""
        try:
            # Get all descendant leaves for this component
            if component_id in graph.components:
                component = graph.components[component_id]
                if hasattr(component, 'get_all_descendants'):
                    return component.get_all_descendants()
                elif hasattr(component, 'get_all_children'):
                    # Recursively get all descendants
                    descendants = []
                    to_visit = list(component.get_all_children())
                    visited = set()
                    
                    while to_visit:
                        child_id = to_visit.pop(0)
                        if child_id in visited:
                            continue
                        visited.add(child_id)
                        descendants.append(child_id)
                        
                        if child_id in graph.components:
                            child_component = graph.components[child_id]
                            if hasattr(child_component, 'get_all_children'):
                                to_visit.extend(child_component.get_all_children())
                    
                    return descendants
            return []
        except Exception as e:
            self.logger.warning(f"Could not determine descendants for {component_id}: {e}")
            return []
    
    def _calculate_propagated_changes(
        self,
        original_graph: 'PortfolioGraph',
        modified_graph: 'PortfolioGraph',
        affected_components: List[str]
    ) -> Dict[str, WeightChange]:
        """Calculate weight changes propagated to descendant components"""
        propagated_changes = {}
        
        for comp_id in affected_components:
            try:
                # Get original weights
                orig_port_weight = self._get_component_weight(original_graph, comp_id, 'portfolio_weight')
                orig_bench_weight = self._get_component_weight(original_graph, comp_id, 'benchmark_weight')
                
                # Get modified weights
                mod_port_weight = self._get_component_weight(modified_graph, comp_id, 'portfolio_weight')
                mod_bench_weight = self._get_component_weight(modified_graph, comp_id, 'benchmark_weight')
                
                # Create weight change if there's a difference
                if (abs(orig_port_weight - mod_port_weight) > 1e-10 or 
                    abs(orig_bench_weight - mod_bench_weight) > 1e-10):
                    
                    propagated_changes[comp_id] = WeightChange(
                        component_id=comp_id,
                        portfolio_weight_before=orig_port_weight,
                        portfolio_weight_after=mod_port_weight,
                        benchmark_weight_before=orig_bench_weight,
                        benchmark_weight_after=mod_bench_weight
                    )
                    
            except Exception as e:
                self.logger.warning(f"Could not calculate propagated change for {comp_id}: {e}")
                continue
        
        return propagated_changes
    
    def _get_component_weight(
        self,
        graph: 'PortfolioGraph',
        component_id: str,
        weight_type: str
    ) -> float:
        """Get weight for a component from the graph"""
        try:
            metric = graph.metric_store.get_metric(component_id, weight_type)
            if metric is not None:
                return metric.value()
            return 0.0
        except Exception:
            return 0.0
    
    def _compute_hierarchical_attribution(
        self,
        baseline_context: 'HierarchicalModelContext',
        scenario_context: 'HierarchicalModelContext',
        decision: PortfolioDecision,
        all_weight_changes: Dict[str, WeightChange]
    ) -> DecisionRiskAttribution:
        """
        Compute comprehensive hierarchical risk attribution.
        
        Parameters
        ----------
        baseline_context : HierarchicalModelContext
            Risk context before decision
        scenario_context : HierarchicalModelContext
            Risk context after decision
        decision : PortfolioDecision
            Original decision
        all_weight_changes : Dict[str, WeightChange]
            All weight changes (direct + propagated)
            
        Returns
        -------
        DecisionRiskAttribution
            Complete attribution results
        """
        attribution = DecisionRiskAttribution(decision)
        
        # Overall risk impact
        baseline_decomposer = baseline_context.active_decomposer
        scenario_decomposer = scenario_context.active_decomposer
        
        if baseline_decomposer is not None and scenario_decomposer is not None:
            attribution.total_active_risk_before = baseline_decomposer.total_active_risk or 0.0
            attribution.total_active_risk_after = scenario_decomposer.total_active_risk or 0.0
            
            # Risk decomposition changes
            attribution.allocation_factor_risk_change = (
                (scenario_decomposer.allocation_factor_risk or 0.0) - 
                (baseline_decomposer.allocation_factor_risk or 0.0)
            )
            attribution.allocation_specific_risk_change = (
                (scenario_decomposer.allocation_specific_risk or 0.0) - 
                (baseline_decomposer.allocation_specific_risk or 0.0)
            )
            attribution.selection_factor_risk_change = (
                (scenario_decomposer.selection_factor_risk or 0.0) - 
                (baseline_decomposer.selection_factor_risk or 0.0)
            )
            attribution.selection_specific_risk_change = (
                (scenario_decomposer.selection_specific_risk or 0.0) - 
                (baseline_decomposer.selection_specific_risk or 0.0)
            )
            
            # Factor exposure changes
            if (baseline_decomposer.active_factor_exposure is not None and 
                scenario_decomposer.active_factor_exposure is not None):
                
                baseline_factors = baseline_decomposer.active_factor_exposure
                scenario_factors = scenario_decomposer.active_factor_exposure
                
                if hasattr(baseline_context, 'get_factor_names'):
                    factor_names = baseline_context.get_factor_names()
                    for i, factor_name in enumerate(factor_names):
                        if i < len(baseline_factors) and i < len(scenario_factors):
                            attribution.factor_exposure_changes[factor_name] = (
                                scenario_factors[i] - baseline_factors[i]
                            )
            
            # Component-level attributions
            self._compute_component_attributions(
                attribution, baseline_context, scenario_context, all_weight_changes
            )
        
        return attribution
    
    def _compute_component_attributions(
        self,
        attribution: DecisionRiskAttribution,
        baseline_context: 'HierarchicalModelContext',
        scenario_context: 'HierarchicalModelContext',
        all_weight_changes: Dict[str, WeightChange]
    ) -> None:
        """Compute component-level risk attributions"""
        baseline_decomposer = baseline_context.active_decomposer
        scenario_decomposer = scenario_context.active_decomposer
        
        if (baseline_decomposer is None or scenario_decomposer is None or
            baseline_decomposer.asset_names is None or scenario_decomposer.asset_names is None):
            return
        
        # Get asset names (these should be leaf component IDs)
        asset_names = baseline_decomposer.asset_names
        
        for i, asset_name in enumerate(asset_names):
            if asset_name not in all_weight_changes:
                continue
                
            weight_change = all_weight_changes[asset_name]
            
            # Get risk contributions before and after
            baseline_total = 0.0
            scenario_total = 0.0 
            
            if (baseline_decomposer.asset_total_contributions is not None and
                i < len(baseline_decomposer.asset_total_contributions)):
                baseline_total = baseline_decomposer.asset_total_contributions[i]
            
            if (scenario_decomposer.asset_total_contributions is not None and
                i < len(scenario_decomposer.asset_total_contributions)):
                scenario_total = scenario_decomposer.asset_total_contributions[i]
            
            # Get component-specific changes
            alloc_factor_change = 0.0
            alloc_specific_change = 0.0
            select_factor_change = 0.0
            select_specific_change = 0.0
            
            if (baseline_decomposer.asset_allocation_factor_contributions is not None and
                scenario_decomposer.asset_allocation_factor_contributions is not None and
                i < len(baseline_decomposer.asset_allocation_factor_contributions) and
                i < len(scenario_decomposer.asset_allocation_factor_contributions)):
                alloc_factor_change = (
                    scenario_decomposer.asset_allocation_factor_contributions[i] - 
                    baseline_decomposer.asset_allocation_factor_contributions[i]
                )
            
            if (baseline_decomposer.asset_allocation_specific_contributions is not None and
                scenario_decomposer.asset_allocation_specific_contributions is not None and
                i < len(baseline_decomposer.asset_allocation_specific_contributions) and
                i < len(scenario_decomposer.asset_allocation_specific_contributions)):
                alloc_specific_change = (
                    scenario_decomposer.asset_allocation_specific_contributions[i] - 
                    baseline_decomposer.asset_allocation_specific_contributions[i]
                )
            
            if (baseline_decomposer.asset_selection_factor_contributions is not None and
                scenario_decomposer.asset_selection_factor_contributions is not None and
                i < len(baseline_decomposer.asset_selection_factor_contributions) and
                i < len(scenario_decomposer.asset_selection_factor_contributions)):
                select_factor_change = (
                    scenario_decomposer.asset_selection_factor_contributions[i] - 
                    baseline_decomposer.asset_selection_factor_contributions[i]
                )
            
            if (baseline_decomposer.asset_selection_specific_contributions is not None and
                scenario_decomposer.asset_selection_specific_contributions is not None and
                i < len(baseline_decomposer.asset_selection_specific_contributions) and
                i < len(scenario_decomposer.asset_selection_specific_contributions)):
                select_specific_change = (
                    scenario_decomposer.asset_selection_specific_contributions[i] - 
                    baseline_decomposer.asset_selection_specific_contributions[i]
                )
            
            # Create component attribution
            comp_attribution = ComponentRiskAttribution(
                component_id=asset_name,
                component_type='leaf',  # Assets are leaves
                weight_change=weight_change,
                total_risk_contribution_before=baseline_total,
                total_risk_contribution_after=scenario_total,
                allocation_factor_contribution_change=alloc_factor_change,
                allocation_specific_contribution_change=alloc_specific_change,
                selection_factor_contribution_change=select_factor_change,
                selection_specific_contribution_change=select_specific_change,
                factor_exposure_changes={}  # Would be computed if needed
            )
            
            attribution.component_attributions[asset_name] = comp_attribution


# Utility functions for decision creation
def create_equity_bonds_tilt(
    equity_tilt: float,
    current_equity_weight: float = 0.6,
    current_bond_weight: float = 0.4,
    description: Optional[str] = None
) -> PortfolioDecision:
    """
    Create a decision representing an equity vs bonds tilt.
    
    Parameters
    ----------
    equity_tilt : float
        Amount to tilt equity (positive = overweight equity)
    current_equity_weight : float, default 0.6
        Current equity weight
    current_bond_weight : float, default 0.4
        Current bond weight
    description : str, optional
        Decision description
        
    Returns
    -------
    PortfolioDecision
        Decision representing the equity/bonds tilt
    """
    desc = description or f"Tilt equity vs bonds by {equity_tilt:+.2%}"
    
    # Create weight changes for both equity and bonds
    equity_change = WeightChange(
        component_id='equity',
        portfolio_weight_before=current_equity_weight,
        portfolio_weight_after=current_equity_weight + equity_tilt,
        benchmark_weight_before=current_equity_weight,  # Assume benchmark unchanged
        benchmark_weight_after=current_equity_weight
    )
    
    bonds_change = WeightChange(
        component_id='bonds',
        portfolio_weight_before=current_bond_weight,
        portfolio_weight_after=current_bond_weight - equity_tilt,  # Offsetting change
        benchmark_weight_before=current_bond_weight,
        benchmark_weight_after=current_bond_weight
    )
    
    return PortfolioDecision(
        component_id='equity',  # Primary decision component
        decision_type=DecisionType.TILT,
        weight_changes={
            'equity': equity_change,
            'bonds': bonds_change
        },
        description=desc
    )