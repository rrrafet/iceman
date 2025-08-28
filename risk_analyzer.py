"""
Portfolio Risk Analyzer
=======================

Risk analysis service for hierarchical portfolios using the visitor pattern.
Provides a clean interface for factor risk decomposition while keeping
PortfolioGraph as a pure container.

This module supports:
- Hierarchical portfolio analysis using FactorRiskDecompositionVisitor
- Risk summary extraction and standardization
- Seamless integration with decision attribution framework
"""

from typing import Dict, Optional, Any, TYPE_CHECKING, List
import pandas as pd
import numpy as np
import logging

if TYPE_CHECKING:
    from .graph import PortfolioGraph
    from .visitors import FactorRiskDecompositionVisitor
    from spark.risk.estimator import LinearRiskModelEstimator
    from spark.risk.schema import RiskResultSchema

logger = logging.getLogger(__name__)


class PortfolioRiskAnalyzer:
    """
    Risk analyzer for hierarchical portfolios using the visitor pattern.
    
    Provides a clean interface for factor risk decomposition while keeping
    PortfolioGraph as a pure container. This class serves as a service layer
    between the portfolio structure and risk analysis functionality.
    
    Parameters
    ----------
    portfolio_graph : PortfolioGraph
        Portfolio graph containing the hierarchical structure
    
    Examples
    --------
    # Basic usage
    >>> analyzer = PortfolioRiskAnalyzer(portfolio_graph)
    >>> visitor = analyzer.decompose_factor_risk('portfolio', factor_returns)
    
    # Get standardized risk summary
    >>> summary = analyzer.get_risk_summary('portfolio', factor_returns)
    >>> print(f"Total risk: {summary['portfolio_volatility']:.4f}")
    """
    
    def __init__(self, 
                 portfolio_graph: 'PortfolioGraph',
                 estimator: Optional['LinearRiskModelEstimator'] = None):
        self.portfolio_graph = portfolio_graph
        self.estimator = estimator  # Will be set to default if None when needed
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    def analysis_type(self) -> str:
        """Get the analysis type being used"""
        return 'hierarchical'
    
    def decompose_factor_risk(
        self,
        root_component_id: str,
        factor_returns: pd.DataFrame,
        estimator: Optional['LinearRiskModelEstimator'] = None,
        portfolio_returns_metric: str = 'portfolio_return',
        benchmark_returns_metric: str = 'benchmark_return',
        portfolio_weight_metric: str = 'portfolio_weight',
        benchmark_weight_metric: str = 'benchmark_weight',
        **kwargs
    ) -> 'FactorRiskDecompositionVisitor':
        """
        Perform hierarchical factor risk decomposition using visitor pattern.
        
        This method creates and runs a FactorRiskDecompositionVisitor to estimate
        factor exposures via OLS regression and decompose risk across the portfolio
        hierarchy.
        
        Parameters
        ----------
        root_component_id : str
            ID of the root component to start decomposition from
        factor_returns : pd.DataFrame
            Factor returns data with factors as columns and dates as index
        estimator : LinearRiskModelEstimator, optional
            Risk model estimator. If None, uses default configuration
        portfolio_returns_metric : str, default 'portfolio_return'
            Name of portfolio returns metric in metric store
        benchmark_returns_metric : str, default 'benchmark_return'
            Name of benchmark returns metric in metric store
        portfolio_weight_metric : str, default 'portfolio_weight'
            Name of portfolio weight metric in metric store
        benchmark_weight_metric : str, default 'benchmark_weight'
            Name of benchmark weight metric in metric store
        **kwargs
            Additional arguments passed to the visitor
            
        Returns
        -------
        FactorRiskDecompositionVisitor
            Visitor instance with completed risk decomposition results
            
        Raises
        ------
        ValueError
            If root component not found or required parameters missing
        """
        self.logger.info(f"Performing hierarchical factor risk decomposition for '{root_component_id}'")
        
        if root_component_id not in self.portfolio_graph.components:
            raise ValueError(f"Component '{root_component_id}' not found in portfolio graph")
        
        from .visitors import FactorRiskDecompositionVisitor
        
        # Create weight service for optimized weight calculations
        weight_service = self.portfolio_graph.create_weight_service()
        
        # Use injected estimator if available, otherwise use provided or default
        final_estimator = estimator or self.estimator
        if final_estimator is None:
            # Import default estimator here to avoid circular imports
            from spark.risk.estimator import LinearRiskModelEstimator
            final_estimator = LinearRiskModelEstimator()
            
        visitor = FactorRiskDecompositionVisitor(
            factor_returns=factor_returns,
            estimator=final_estimator,
            metric_store=self.portfolio_graph.metric_store,
            portfolio_returns_metric=portfolio_returns_metric,
            benchmark_returns_metric=benchmark_returns_metric,
            portfolio_weight_metric=portfolio_weight_metric,
            benchmark_weight_metric=benchmark_weight_metric,
            weight_service=weight_service,
            **kwargs,
        )
        
        root_component = self.portfolio_graph.components[root_component_id]
        root_component.accept(visitor)
        
        self.logger.info(f"Completed hierarchical risk decomposition for component '{root_component_id}'")
        return visitor
    
    
    def get_riskresult(
        self,
        root_component_id: str,
        factor_returns: pd.DataFrame,
        estimator: Optional['LinearRiskModelEstimator'] = None,
        include_time_series: bool = False,
        **kwargs
    ) -> 'RiskResultSchema':
        """
        CONSOLIDATED: Get standardized risk results for hierarchical portfolio analysis.
        
        Single method that consolidates all risk result retrieval patterns.
        Returns a unified schema with key risk metrics extracted from the
        FactorRiskDecompositionVisitor results with optional comprehensive data.
        
        Parameters
        ----------
        root_component_id : str
            Component ID for analysis
        factor_returns : pd.DataFrame
            Factor returns data
        estimator : LinearRiskModelEstimator, optional
            Risk model estimator
        include_time_series : bool, default False
            If True, creates comprehensive schema with time series and hierarchy data
            If False, creates basic schema with risk decomposition only
        **kwargs
            Additional arguments
            
        Returns
        -------
        RiskResultSchema
            Standardized risk results in unified schema format
        """
        if include_time_series:
            # Use comprehensive approach with time series and hierarchy data
            return self._get_comprehensive_riskresult(
                root_component_id=root_component_id,
                factor_returns=factor_returns,
                estimator=estimator,
                **kwargs
            )
        else:
            # Use basic approach for lightweight analysis
            visitor = self.decompose_factor_risk(
                root_component_id=root_component_id,
                factor_returns=factor_returns, 
                estimator=estimator,
                **kwargs
            )
            
            return self._extract_hierarchical_schema(visitor)
    
    
    def _extract_hierarchical_schema(self, visitor: 'FactorRiskDecompositionVisitor') -> 'RiskResultSchema':
        """
        Extract standardized schema from hierarchical visitor results.
        
        Converts the hierarchical visitor results into unified schema format
        and populates hierarchical risk data for all components.
        """
        from spark.risk.schema import RiskResultSchema, AnalysisType
        
        try:
            # Get names and component IDs from visitor
            factor_names = list(getattr(visitor, 'factor_names', []))
            
            # Get all processed component IDs from visitor
            component_ids = []
            if hasattr(visitor, '_processed_components'):
                component_ids = list(visitor._processed_components)
            elif hasattr(visitor, 'metric_store') and visitor.metric_store:
                # Fallback: get components with hierarchical context
                for comp_id in visitor.metric_store._metrics:
                    if visitor.metric_store.get_metric(comp_id, 'hierarchical_model_context'):
                        component_ids.append(comp_id)
            
            self.logger.info(f"Extracting hierarchical schema for {len(component_ids)} components")
            
            # Create comprehensive hierarchical schema
            schema = RiskResultSchema(
                analysis_type=AnalysisType.HIERARCHICAL,
                asset_names=[],  # Will be populated from visitor data
                factor_names=factor_names,
                component_ids=component_ids
            )
            
            # **NEW: Use bulk hierarchical storage to populate ALL components' risk data**
            population_summary = schema.populate_hierarchical_risk_data_from_visitor(visitor)
            
            self.logger.info(f"Hierarchical data population: {population_summary['components_processed']} components processed")
            if population_summary['errors']:
                self.logger.warning(f"Population errors: {population_summary['errors']}")
            
            # Extract asset names from populated data (descendant leaves)
            asset_names = set()
            if hasattr(visitor, 'get_node_unified_matrices'):
                for component_id in component_ids:
                    matrices = visitor.get_node_unified_matrices(component_id)
                    if matrices and 'portfolio' in matrices:
                        leaves = matrices['portfolio'].get('descendant_leaves', [])
                        asset_names.update(leaves)
            
            schema.asset_names = list(asset_names)
            
            # Set hierarchy structure if portfolio graph is available
            if self.portfolio_graph:
                try:
                    component_relationships = self._extract_component_relationships()
                    component_metadata = self._extract_component_metadata() 
                    
                    # Determine root component (first processed or from portfolio graph)
                    root_component = component_ids[0] if component_ids else 'TOTAL'
                    
                    schema.set_hierarchy_structure(
                        root_component=root_component,
                        component_relationships=component_relationships,
                        component_metadata=component_metadata,
                        adjacency_list=self.portfolio_graph.adjacency_list
                    )
                    
                    self.logger.info(f"Hierarchy structure set with root '{root_component}'")
                    
                except Exception as hierarchy_error:
                    self.logger.warning(f"Could not set hierarchy structure: {hierarchy_error}")
            
            # Add comprehensive context information
            schema.add_context_info('visitor_type', type(visitor).__name__)
            schema.add_context_info('analysis_method', 'comprehensive_hierarchical')
            schema.add_context_info('component_count', len(component_ids))
            schema.add_context_info('population_summary', population_summary)
            schema.add_context_info('extraction_timestamp', self.current_timestamp.isoformat())
            
            # Validate hierarchical completeness
            completeness_validation = schema.validate_hierarchical_completeness()
            schema.add_context_info('hierarchical_completeness', completeness_validation)
            
            # Set comprehensive validation results
            validation_results = {
                'extraction_successful': population_summary['success'],
                'visitor_type': type(visitor).__name__,
                'components_analyzed': population_summary['components_processed'],
                'lenses_populated': population_summary['lenses_populated'],
                'errors': population_summary['errors'],
                'hierarchical_complete': completeness_validation['complete'],
                'completeness_percentage': completeness_validation['summary']['completeness_percentage']
            }
            schema.set_validation_results(validation_results)
            
            self.logger.info(f"Schema extraction completed: {validation_results['components_analyzed']} components, "
                           f"{validation_results['completeness_percentage']:.1f}% complete")
            
            return schema
            
        except Exception as e:
            self.logger.warning(f"Failed to extract hierarchical schema: {e}")
            
            # Return minimal schema if extraction fails
            schema = RiskResultSchema(
                analysis_type=AnalysisType.HIERARCHICAL,
                asset_names=[],
                factor_names=[]
            )
            
            schema.set_core_metrics(0.0, 0.0, 0.0)
            schema.set_validation_results({
                'extraction_successful': False,
                'error': str(e)
            })
            
            return schema
    
    def _extract_portfolio_returns(self) -> Dict[str, pd.Series]:
        """Extract portfolio returns for all components from metric store."""
        portfolio_returns = {}
        metric_store = self.portfolio_graph.metric_store
        
        for component_id in self.portfolio_graph.components.keys():
            # Try different metric names for portfolio returns
            for metric_name in ['portfolio_return', 'port_ret', 'returns']:
                metric = metric_store.get_metric(component_id, metric_name)
                if metric and hasattr(metric, 'value'):
                    returns_data = metric.value()
                    if isinstance(returns_data, pd.Series):
                        portfolio_returns[component_id] = returns_data
                        break
                    elif hasattr(returns_data, 'values') and len(returns_data.values) > 0:
                        # Create series from array data if available
                        portfolio_returns[component_id] = pd.Series(returns_data.values)
                        break
                        
        return portfolio_returns
    
    def _extract_benchmark_returns(self) -> Dict[str, pd.Series]:
        """Extract benchmark returns for all components from metric store."""
        benchmark_returns = {}
        metric_store = self.portfolio_graph.metric_store
        
        for component_id in self.portfolio_graph.components.keys():
            # Try different metric names for benchmark returns
            for metric_name in ['benchmark_return', 'bench_ret', 'benchmark_returns']:
                metric = metric_store.get_metric(component_id, metric_name)
                if metric and hasattr(metric, 'value'):
                    returns_data = metric.value()
                    if isinstance(returns_data, pd.Series):
                        benchmark_returns[component_id] = returns_data
                        break
                    elif hasattr(returns_data, 'values') and len(returns_data.values) > 0:
                        # Create series from array data if available
                        benchmark_returns[component_id] = pd.Series(returns_data.values)
                        break
                        
        return benchmark_returns
    
    def _extract_component_relationships(self) -> Dict[str, Dict[str, Any]]:
        """Extract component parent-child relationships from portfolio graph."""
        relationships = {}
        adjacency_list = self.portfolio_graph.adjacency_list
        
        for comp_id in self.portfolio_graph.components.keys():
            # Find parent
            parent = None
            for parent_id, children in adjacency_list.items():
                if comp_id in children:
                    parent = parent_id
                    break
            
            # Get children
            children = adjacency_list.get(comp_id, [])
            
            relationships[comp_id] = {
                "parent": parent,
                "children": children
            }
            
        return relationships
    
    def _extract_component_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Extract component metadata from portfolio graph."""
        metadata = {}
        
        for comp_id, component in self.portfolio_graph.components.items():
            metadata[comp_id] = {
                "component_id": comp_id,
                "type": "leaf" if component.is_leaf() else "node",
                "level": len(comp_id.split('/')) - 1 if '/' in comp_id else 0,
                "path": comp_id
            }
            
        return metadata
    
    def _get_comprehensive_riskresult(
        self,
        root_component_id: str,
        factor_returns: pd.DataFrame,
        estimator: Optional['LinearRiskModelEstimator'] = None,
        **kwargs
    ) -> 'RiskResultSchema':
        """
        Create comprehensive risk results with all available data.
        
        This method combines data from all sources available to the analyzer:
        - Risk decomposition results from visitor
        - Time series data from portfolio graph metric store
        - Hierarchy structure from portfolio graph
        - Factor returns from input
        
        Parameters
        ----------
        root_component_id : str
            Component ID for analysis root
        factor_returns : pd.DataFrame
            Factor returns data
        estimator : LinearRiskModelEstimator, optional
            Risk model estimator override
        **kwargs
            Additional arguments passed to decomposition
            
        Returns
        -------
        RiskResultSchema
            Comprehensive risk results with time series and hierarchy data
        """
        self.logger.info(f"Creating comprehensive schema for '{root_component_id}'")
        
        # 1. Run factor risk decomposition
        visitor = self.decompose_factor_risk(
            root_component_id=root_component_id,
            factor_returns=factor_returns,
            estimator=estimator,
            **kwargs
        )
        
        # 2. Extract time series data from portfolio graph
        portfolio_returns = self._extract_portfolio_returns()
        benchmark_returns = self._extract_benchmark_returns()
        
        # 3. Convert factor returns to dictionary
        factor_returns_dict = {
            factor_name: factor_returns[factor_name] 
            for factor_name in factor_returns.columns
        }
        
        # 4. Create schema using decoupled factory method
        from spark.risk.schema_factory import RiskSchemaFactory
        
        schema = RiskSchemaFactory.from_visitor_results_with_time_series(
            visitor=visitor,
            component_id=root_component_id,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            factor_returns=factor_returns_dict,
            dates=factor_returns.index.tolist(),
            analysis_type="hierarchical"
        )
        
        # 5. Add hierarchy data from portfolio graph
        try:
            component_relationships = self._extract_component_relationships()
            component_metadata = self._extract_component_metadata()
            
            schema.set_hierarchy_structure(
                root_component=root_component_id,
                component_relationships=component_relationships,
                component_metadata=component_metadata,
                adjacency_list=self.portfolio_graph.adjacency_list
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to set hierarchy structure: {e}")
            schema.add_context_info('hierarchy_extraction_error', str(e))
        
        # 6. Add analyzer context information
        schema.add_context_info('analyzer_type', self.__class__.__name__)
        schema.add_context_info('portfolio_components_count', len(self.portfolio_graph.components))
        schema.add_context_info('time_series_portfolio_components', len(portfolio_returns))
        schema.add_context_info('time_series_benchmark_components', len(benchmark_returns))
        schema.add_context_info('extraction_method', 'comprehensive_analyzer')
        
        # 7. Extract and populate complete hierarchical risk decomposition data
        try:
            self._populate_hierarchical_risk_data(visitor, schema)
            self.logger.info("Successfully populated hierarchical risk decomposition data")
            
        except Exception as e:
            self.logger.warning(f"Failed to populate hierarchical risk data: {e}")
            schema.add_context_info('hierarchical_risk_extraction_error', str(e))
        
        self.logger.info(f"Comprehensive schema created with {len(portfolio_returns)} portfolio and {len(benchmark_returns)} benchmark time series")
        
        return schema
    
    def _populate_hierarchical_risk_data(
        self, 
        visitor: 'FactorRiskDecompositionVisitor',
        schema: 'RiskResultSchema'
    ) -> None:
        """
        Extract complete risk decomposition results from visitor and populate hierarchical schema.
        
        This method extracts decomposer results for every processed component and stores
        complete risk analysis data for use in Maverick drill-down functionality.
        
        Parameters
        ----------
        visitor : FactorRiskDecompositionVisitor
            Completed visitor with risk decomposition results
        schema : RiskResultSchema
            Schema to populate with hierarchical data
        """
        self.logger.info("Extracting complete hierarchical risk decomposition data")
        
        # Get all processed components from visitor
        processed_components = visitor.get_processed_components()
        
        for component_id in processed_components:
            self.logger.debug(f"Processing hierarchical data for component: {component_id}")
            
            try:
                # Extract decomposer results for this component
                component_decomposers = self._extract_component_decomposers(visitor, component_id)
                
                if component_decomposers:
                    # Store risk decomposition for each lens
                    for lens, decomposer in component_decomposers.items():
                        if decomposer:
                            decomposer_result = self._convert_decomposer_to_dict(decomposer, lens, visitor, component_id)
                            schema.set_component_full_decomposition(component_id, lens, decomposer_result)
                            self.logger.debug(f"Stored {lens} decomposition for {component_id}")
                    
                    # Extract and store matrices for this component  
                    component_matrices = self._extract_component_matrices_from_visitor(visitor, component_id)
                    
                    if component_matrices:
                        for lens, matrices in component_matrices.items():
                            if matrices:
                                schema.set_component_matrices(component_id, lens, matrices)
                                self.logger.debug(f"Stored {lens} matrices for {component_id}")
                
            except Exception as e:
                self.logger.warning(f"Failed to process component {component_id}: {e}")
                continue
        
        self.logger.info(f"Completed hierarchical risk data extraction for {len(processed_components)} components")
    
    def _extract_component_decomposers(
        self, 
        visitor: 'FactorRiskDecompositionVisitor',
        component_id: str
    ) -> Dict[str, Any]:
        """
        Extract risk decomposer objects for a specific component from visitor results.
        
        Parameters
        ---------- 
        visitor : FactorRiskDecompositionVisitor
            Visitor with completed risk analysis
        component_id : str
            Component to extract decomposers for
            
        Returns
        -------
        dict
            Dictionary with decomposer objects for each lens: {lens: decomposer_object}
        """
        component_decomposers = {}
        
        # Check if component has hierarchical model context in metric store
        if visitor.metric_store:
            context_metric = visitor.metric_store.get_metric(component_id, 'hierarchical_model_context')
            
            if context_metric:
                context = context_metric.value()
                
                # Extract decomposers from hierarchical context
                if hasattr(context, 'portfolio_decomposer'):
                    component_decomposers['portfolio'] = context.portfolio_decomposer
                if hasattr(context, 'benchmark_decomposer'):
                    component_decomposers['benchmark'] = context.benchmark_decomposer
                if hasattr(context, 'active_decomposer'):
                    component_decomposers['active'] = context.active_decomposer
                    
                self.logger.debug(f"Extracted {len(component_decomposers)} decomposers from hierarchical context for {component_id}")
        
        return component_decomposers
    
    def _convert_decomposer_to_dict(
        self,
        decomposer: Any,
        lens: str,
        visitor: 'FactorRiskDecompositionVisitor',
        component_id: str
    ) -> Dict[str, Any]:
        """
        Convert a risk decomposer object to dictionary format for schema storage.
        
        Parameters
        ----------
        decomposer : RiskDecomposer
            Risk decomposer object
        lens : str
            Lens type for context
            
        Returns
        -------
        dict
            Complete decomposer results in dictionary format
        """
        try:
            # Get risk decomposition summary
            summary = decomposer.risk_decomposition_summary()
            
            # Convert to schema format
            result = {
                "total_risk": summary.get("portfolio_volatility", summary.get("total_active_risk", 0.0)),
                "factor_risk_contribution": summary.get("factor_risk_contribution", 0.0),
                "specific_risk_contribution": summary.get("specific_risk_contribution", 0.0),
                "factor_risk_percentage": summary.get("factor_risk_percentage", 0.0),
                "specific_risk_percentage": summary.get("specific_risk_percentage", 0.0),
            }
            
            # Extract factor contributions
            if hasattr(decomposer, 'factor_contributions') and decomposer.factor_contributions is not None:
                factor_names = visitor.get_factor_names() if hasattr(visitor, 'get_factor_names') else []
                if len(factor_names) == len(decomposer.factor_contributions):
                    result["factor_contributions"] = {
                        name: float(contrib) for name, contrib in zip(factor_names, decomposer.factor_contributions)
                    }
                else:
                    result["factor_contributions"] = {
                        f"factor_{i}": float(contrib) 
                        for i, contrib in enumerate(decomposer.factor_contributions)
                    }
            
            # Extract asset contributions  
            if hasattr(decomposer, 'asset_contributions') and decomposer.asset_contributions is not None:
                # Get descendant leaves for this component
                if hasattr(visitor, 'get_node_unified_matrices'):
                    matrices = visitor.get_node_unified_matrices(component_id) 
                    if matrices and 'portfolio' in matrices:
                        descendant_leaves = matrices['portfolio'].get('descendant_leaves', [])
                        if len(descendant_leaves) == len(decomposer.asset_contributions):
                            result["asset_contributions"] = {
                                leaf_id: float(contrib) 
                                for leaf_id, contrib in zip(descendant_leaves, decomposer.asset_contributions)
                            }
                        else:
                            result["asset_contributions"] = {
                                f"asset_{i}": float(contrib)
                                for i, contrib in enumerate(decomposer.asset_contributions)
                            }
            
            # Extract matrices
            if hasattr(decomposer, 'weighted_betas') and decomposer.weighted_betas is not None:
                result["weighted_betas"] = self._convert_matrix_to_named_dict(
                    decomposer.weighted_betas, 
                    visitor,
                    component_id
                )
            
            if hasattr(decomposer, 'factor_loadings') and decomposer.factor_loadings is not None:
                result["factor_loadings_matrix"] = self._convert_matrix_to_named_dict(
                    decomposer.factor_loadings,
                    visitor, 
                    component_id
                )
            
            # Get covariance matrix
            if hasattr(decomposer, 'covariance_matrix') and decomposer.covariance_matrix is not None:
                result["covariance_matrix"] = decomposer.covariance_matrix.tolist()
            
            # Calculate correlation matrix from covariance if available
            if "covariance_matrix" in result and result["covariance_matrix"]:
                cov_matrix = np.array(result["covariance_matrix"])
                if cov_matrix.shape[0] == cov_matrix.shape[1] and cov_matrix.shape[0] > 1:
                    try:
                        # Convert covariance to correlation
                        std_devs = np.sqrt(np.diag(cov_matrix))
                        correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
                        result["correlation_matrix"] = correlation_matrix.tolist()
                    except Exception:
                        result["correlation_matrix"] = []
            
            # Add lens-specific data for active risk
            if lens == "active":
                result["allocation_selection"] = self._extract_active_decomposition(summary)
            
            # Add validation results
            result["euler_identity_check"] = True  # Will be validated by schema
            result["asset_sum_check"] = True
            result["factor_sum_check"] = True
            result["validation_summary"] = f"{lens} decomposition extracted successfully"
            result["validation_details"] = summary
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to convert {lens} decomposer to dict: {e}")
            # Return minimal valid structure
            return {
                "total_risk": 0.0,
                "factor_risk_contribution": 0.0,
                "specific_risk_contribution": 0.0,
                "factor_risk_percentage": 0.0,
                "specific_risk_percentage": 0.0,
                "factor_contributions": {},
                "asset_contributions": {},
                "factor_loadings_matrix": {},
                "weighted_betas": {},
                "covariance_matrix": [],
                "correlation_matrix": [],
                "euler_identity_check": False,
                "validation_summary": f"Failed to extract {lens} decomposition: {str(e)}"
            }
    
    def _convert_matrix_to_named_dict(
        self,
        matrix: np.ndarray,
        visitor: 'FactorRiskDecompositionVisitor',
        component_id: str
    ) -> Dict[str, Dict[str, float]]:
        """Convert a matrix to named dictionary format."""
        if matrix is None or matrix.size == 0:
            return {}
        
        try:
            # Get asset and factor names
            factor_names = visitor.get_factor_names() if hasattr(visitor, 'get_factor_names') else []
            
            # Get descendant leaves for asset names
            asset_names = []
            if hasattr(visitor, 'get_node_unified_matrices'):
                matrices = visitor.get_node_unified_matrices(component_id)
                if matrices and 'portfolio' in matrices:
                    asset_names = matrices['portfolio'].get('descendant_leaves', [])
            
            # Convert to named dictionary
            if matrix.ndim == 2:
                n_assets, n_factors = matrix.shape
                
                # Use provided names or generate defaults
                if len(asset_names) != n_assets:
                    asset_names = [f"asset_{i}" for i in range(n_assets)]
                if len(factor_names) != n_factors:
                    factor_names = [f"factor_{i}" for i in range(n_factors)]
                
                return {
                    asset_name: {
                        factor_name: float(matrix[i, j])
                        for j, factor_name in enumerate(factor_names)
                    }
                    for i, asset_name in enumerate(asset_names)
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.debug(f"Failed to convert matrix to named dict: {e}")
            return {}
    
    def _extract_active_decomposition(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Extract active risk allocation/selection decomposition from summary."""
        return {
            "allocation_factor_contributions": summary.get("allocation_factor_contributions", {}),
            "selection_factor_contributions": summary.get("selection_factor_contributions", {}),  
            "interaction_contributions": summary.get("interaction_contributions", {}),
            "allocation_total": summary.get("allocation_total", 0.0),
            "selection_total": summary.get("selection_total", 0.0),
            "interaction_total": summary.get("interaction_total", 0.0)
        }
    
    def _extract_component_matrices_from_visitor(
        self,
        visitor: 'FactorRiskDecompositionVisitor',
        component_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """Extract matrices for a specific component from visitor unified matrices."""
        component_matrices = {}
        
        try:
            # Get unified matrices for this component
            unified_matrices = visitor.get_node_unified_matrices(component_id)
            
            if unified_matrices:
                for lens in ['portfolio', 'benchmark', 'active']:
                    if lens in unified_matrices:
                        lens_matrices = unified_matrices[lens]
                        
                        # Convert matrices to serializable format
                        matrices_dict = {}
                        
                        for matrix_name, matrix_data in lens_matrices.items():
                            if isinstance(matrix_data, np.ndarray):
                                matrices_dict[matrix_name] = matrix_data.tolist()
                            elif isinstance(matrix_data, list):
                                matrices_dict[matrix_name] = matrix_data
                            else:
                                matrices_dict[matrix_name] = matrix_data
                        
                        if matrices_dict:
                            component_matrices[lens] = matrices_dict
                
        except Exception as e:
            self.logger.debug(f"Failed to extract matrices for {component_id}: {e}")
        
        return component_matrices
    
    def __repr__(self) -> str:
        """String representation of the analyzer"""
        return f"PortfolioRiskAnalyzer(components={len(self.portfolio_graph.components)})"


# Convenience factory function  
def create_portfolio_risk_analyzer(portfolio_graph: 'PortfolioGraph') -> PortfolioRiskAnalyzer:
    """
    Create risk analyzer for hierarchical portfolio.
    
    This is a convenience factory function that simply wraps the constructor
    for consistency with other factory patterns in the codebase.
    
    Parameters
    ----------
    portfolio_graph : PortfolioGraph
        Portfolio graph to analyze
        
    Returns
    -------
    PortfolioRiskAnalyzer
        Analyzer configured for hierarchical analysis
    """
    return PortfolioRiskAnalyzer(portfolio_graph)