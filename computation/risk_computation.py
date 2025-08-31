"""
Risk Computation layer for portfolio risk analysis system.
Orchestrates FactorRiskDecompositionVisitor execution and manages computation state.
"""

from typing import Dict, Optional, Any, TYPE_CHECKING
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import time

# Import the visitor from the portfolio module
try:
    from spark.portfolio.visitors import FactorRiskDecompositionVisitor
    from spark.portfolio.graph import PortfolioGraph
    from spark.risk.risk_analysis import RiskResult
except ImportError as e:
    logging.warning(f"Import error: {e}")
    FactorRiskDecompositionVisitor = None
    PortfolioGraph = None
    RiskResult = None

if TYPE_CHECKING:
    from spark.portfolio.visitors import FactorRiskDecompositionVisitor
    from spark.portfolio.graph import PortfolioGraph
    from spark.risk.risk_analysis import RiskResult

logger = logging.getLogger(__name__)


class RiskComputation:
    """
    Manages risk computation execution using FactorRiskDecompositionVisitor.
    
    Provides a clean interface for running risk decomposition and accessing results
    while maintaining computation state and validation.
    """
    
    def __init__(self, portfolio_graph: 'PortfolioGraph', visitor: Optional['FactorRiskDecompositionVisitor'] = None):
        """
        Initialize risk computation manager.
        
        Args:
            portfolio_graph: PortfolioGraph instance
            visitor: Optional existing FactorRiskDecompositionVisitor instance
        """
        if PortfolioGraph is None:
            raise ImportError("PortfolioGraph not available - check spark.portfolio.graph import")
        
        self.portfolio_graph = portfolio_graph
        self._visitor = visitor
        self._computation_timestamp: Optional[datetime] = None
        self._is_computed = False
        self._computation_stats: Dict[str, Any] = {}
        self._last_factor_returns: Optional[pd.DataFrame] = None
        self._is_stale = False
        
        logger.info(f"Initialized RiskComputation for portfolio graph with {len(portfolio_graph.components)} components")
    
    def run_full_decomposition(self, factor_returns: pd.DataFrame) -> bool:
        """
        Run full risk decomposition using FactorRiskDecompositionVisitor.
        
        Args:
            factor_returns: DataFrame with factor returns (wide format expected)
            
        Returns:
            True if computation successful, False otherwise
        """
        if FactorRiskDecompositionVisitor is None:
            logger.error("FactorRiskDecompositionVisitor not available")
            return False
        
        if factor_returns.empty:
            logger.error("Factor returns data is empty")
            return False
        
        try:
            start_time = time.time()
            logger.info("Starting factor risk decomposition...")
            
            # Create visitor if not provided
            if self._visitor is None:
                self._visitor = FactorRiskDecompositionVisitor(
                    factor_returns=factor_returns,
                    metric_store=self.portfolio_graph.metric_store
                )
                logger.debug("Created new FactorRiskDecompositionVisitor")
            
            # Store factor returns for staleness checking
            self._last_factor_returns = factor_returns.copy()
            
            # Run the visitor on the portfolio graph
            # This will populate the metric store with risk results
            root_component_id = self.portfolio_graph.root_id
            if root_component_id and root_component_id in self.portfolio_graph.components:
                root_component = self.portfolio_graph.components[root_component_id]
                
                # Execute visitor traversal
                self.portfolio_graph.accept_visitor(self._visitor, root_component_id)
                
                # Update computation state
                self._computation_timestamp = datetime.now()
                self._is_computed = True
                self._is_stale = False
                
                end_time = time.time()
                computation_time = end_time - start_time
                
                # Store computation statistics
                self._computation_stats = {
                    "computation_time_seconds": computation_time,
                    "nodes_processed": len(self.portfolio_graph.components),
                    "factors_count": factor_returns.shape[1],
                    "factor_returns_shape": factor_returns.shape,
                    "timestamp": self._computation_timestamp.isoformat(),
                    "root_component": root_component_id
                }
                
                logger.info(f"Risk decomposition completed successfully in {computation_time:.2f} seconds")
                return True
            else:
                logger.error(f"Root component not found: {root_component_id}")
                return False
                
        except Exception as e:
            logger.error(f"Risk decomposition failed: {e}")
            self._is_computed = False
            return False
    
    def get_visitor(self) -> Optional['FactorRiskDecompositionVisitor']:
        """
        Get the FactorRiskDecompositionVisitor instance.
        
        Returns:
            FactorRiskDecompositionVisitor instance or None if not available
        """
        return self._visitor
    
    def is_computed(self) -> bool:
        """
        Check if risk decomposition has been computed.
        
        Returns:
            True if computation has been performed successfully
        """
        return self._is_computed
    
    def get_computation_timestamp(self) -> Optional[datetime]:
        """
        Get timestamp of last computation.
        
        Returns:
            Datetime of last computation or None if never computed
        """
        return self._computation_timestamp
    
    def needs_recomputation(self) -> bool:
        """
        Check if recomputation is needed due to staleness.
        
        Returns:
            True if recomputation is needed
        """
        return self._is_stale or not self._is_computed
    
    def mark_stale(self) -> None:
        """Mark the computation as stale, requiring recomputation."""
        self._is_stale = True
        logger.debug("Marked computation as stale")
    
    def get_computation_stats(self) -> Dict[str, Any]:
        """
        Get computation statistics.
        
        Returns:
            Dictionary with computation statistics
        """
        if not self._is_computed:
            return {"computed": False, "message": "No computation performed yet"}
        
        stats = self._computation_stats.copy()
        stats.update({
            "computed": self._is_computed,
            "is_stale": self._is_stale,
            "last_computation": self._computation_timestamp.isoformat() if self._computation_timestamp else None
        })
        
        return stats
    
    def get_node_risk_results(self, component_id: str) -> Dict[str, 'RiskResult']:
        """
        Get risk results for a component across all lenses.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Dictionary mapping lens names to RiskResult objects
        """
        if not self._is_computed:
            logger.warning("Risk computation not performed - no results available")
            return {}
        
        if not self.portfolio_graph.metric_store:
            logger.error("No metric store available")
            return {}
        
        results = {}
        
        # Try to get risk results for different lenses (portfolio, benchmark, active)
        for lens in ['portfolio', 'benchmark', 'active']:
            try:
                # Get from metric store - assuming risk results are stored with lens suffix
                risk_result = self.portfolio_graph.metric_store.get_metric(
                    component_id, f'risk_result_{lens}'
                )
                
                if risk_result and hasattr(risk_result, 'value'):
                    results[lens] = risk_result.value
                elif risk_result:
                    results[lens] = risk_result
                
            except Exception as e:
                logger.debug(f"Could not retrieve {lens} risk result for {component_id}: {e}")
        
        if not results:
            logger.warning(f"No risk results found for component: {component_id}")
        
        return results
    
    def get_risk_result(self, component_id: str, lens: str) -> Optional['RiskResult']:
        """
        Get risk result for a specific component and lens.
        
        Args:
            component_id: Component identifier
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            RiskResult object or None if not found
        """
        if not self._is_computed:
            logger.warning("Risk computation not performed - no results available")
            return None
        
        try:
            risk_result = self.portfolio_graph.metric_store.get_metric(
                component_id, f'risk_result_{lens}'
            )
            
            if risk_result and hasattr(risk_result, 'value'):
                return risk_result.value
            elif risk_result:
                return risk_result
            else:
                logger.debug(f"No {lens} risk result found for component: {component_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving risk result for {component_id}/{lens}: {e}")
            return None
    
    def get_all_risk_results(self) -> Dict[str, Dict[str, 'RiskResult']]:
        """
        Get all risk results for all components and lenses.
        
        Returns:
            Nested dictionary: {component_id: {lens: RiskResult}}
        """
        if not self._is_computed:
            logger.warning("Risk computation not performed - no results available")
            return {}
        
        all_results = {}
        
        for component_id in self.portfolio_graph.components.keys():
            component_results = self.get_node_risk_results(component_id)
            if component_results:
                all_results[component_id] = component_results
        
        logger.debug(f"Retrieved risk results for {len(all_results)} components")
        return all_results
    
    def get_risk_contribution_matrix(self) -> pd.DataFrame:
        """
        Get risk contribution matrix from visitor.
        
        Returns:
            DataFrame with risk contributions or empty DataFrame if not available
        """
        if not self._is_computed or not self._visitor:
            logger.warning("Risk computation not performed or visitor not available")
            return pd.DataFrame()
        
        # Try to extract contribution matrix from visitor
        # This depends on the specific implementation of FactorRiskDecompositionVisitor
        try:
            if hasattr(self._visitor, 'get_contribution_matrix'):
                return self._visitor.get_contribution_matrix()
            elif hasattr(self._visitor, 'contribution_matrix'):
                return self._visitor.contribution_matrix
            else:
                logger.warning("Visitor does not have contribution matrix method/attribute")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error retrieving contribution matrix: {e}")
            return pd.DataFrame()
    
    def validate_risk_decomposition(self, component_id: str) -> Dict[str, Any]:
        """
        Validate risk decomposition for a component (Euler identity check).
        
        Args:
            component_id: Component identifier
            
        Returns:
            Dictionary with validation results
        """
        if not self._is_computed:
            return {"valid": False, "error": "No computation performed"}
        
        validation_result = {
            "component_id": component_id,
            "valid": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Get risk results for the component
            risk_results = self.get_node_risk_results(component_id)
            
            if not risk_results:
                validation_result["errors"].append("No risk results found")
                return validation_result
            
            # Validate each lens
            for lens, risk_result in risk_results.items():
                if risk_result:
                    # Check Euler identity: total_risk² ≈ factor_risk² + specific_risk²
                    total_risk_sq = risk_result.total_risk ** 2
                    factor_specific_sum = risk_result.factor_risk ** 2 + risk_result.specific_risk ** 2
                    
                    relative_error = abs(total_risk_sq - factor_specific_sum) / total_risk_sq if total_risk_sq > 0 else 0
                    
                    if relative_error > 0.01:  # 1% tolerance
                        validation_result["warnings"].append(
                            f"{lens}: Euler identity violation (error: {relative_error:.4f})"
                        )
                    
                    # Check for reasonable values
                    if risk_result.total_risk < 0:
                        validation_result["errors"].append(f"{lens}: Negative total risk")
                    
                    if risk_result.factor_risk < 0 or risk_result.specific_risk < 0:
                        validation_result["errors"].append(f"{lens}: Negative component risks")
            
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def get_computation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive computation summary.
        
        Returns:
            Dictionary with computation summary
        """
        summary = {
            "computed": self._is_computed,
            "stale": self._is_stale,
            "timestamp": self._computation_timestamp.isoformat() if self._computation_timestamp else None,
            "portfolio_components": len(self.portfolio_graph.components),
            "has_visitor": self._visitor is not None,
            "factor_data_available": self._last_factor_returns is not None
        }
        
        if self._is_computed:
            summary.update(self._computation_stats)
            
            # Add results summary
            all_results = self.get_all_risk_results()
            summary["results_summary"] = {
                "components_with_results": len(all_results),
                "total_results": sum(len(lenses) for lenses in all_results.values()),
                "available_lenses": list(set(lens for lenses in all_results.values() for lens in lenses.keys()))
            }
        
        if self._last_factor_returns is not None:
            summary["factor_data_summary"] = {
                "shape": self._last_factor_returns.shape,
                "factors": list(self._last_factor_returns.columns),
                "date_range": (
                    str(self._last_factor_returns.index.min()),
                    str(self._last_factor_returns.index.max())
                ) if hasattr(self._last_factor_returns.index, 'min') else None
            }
        
        return summary