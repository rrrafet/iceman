"""
Risk Analysis Service - main orchestrator for portfolio risk analysis system.
Coordinates data providers, computation, and configuration services.
"""

from typing import Dict, Optional, Any, TYPE_CHECKING
import pandas as pd
import logging
from datetime import datetime

from datamodels import FactorDataProvider, PortfolioDataProvider, RiskModelRegistry
from computation import RiskComputation
from services.configuration_service import ConfigurationService

# Import portfolio graph and related classes
try:
    from spark.portfolio.graph import PortfolioGraph
    from spark.risk.risk_analysis import RiskResult
except ImportError as e:
    logging.warning(f"Import error: {e}")
    PortfolioGraph = None
    RiskResult = None

if TYPE_CHECKING:
    from spark.portfolio.graph import PortfolioGraph
    from spark.risk.risk_analysis import RiskResult

logger = logging.getLogger(__name__)


class RiskAnalysisService:
    """
    Main orchestrator for the portfolio risk analysis system.
    
    Coordinates all system components including data providers, computation engine,
    and configuration management to provide a unified interface for risk analysis.
    """
    
    def __init__(
        self,
        config_service: ConfigurationService,
        factor_provider: FactorDataProvider,
        portfolio_provider: PortfolioDataProvider,
        risk_computation: Optional[RiskComputation] = None
    ):
        """
        Initialize risk analysis service.
        
        Args:
            config_service: Configuration service instance
            factor_provider: Factor data provider instance
            portfolio_provider: Portfolio data provider instance
            risk_computation: Optional risk computation instance
        """
        self.config_service = config_service
        self.factor_provider = factor_provider
        self.portfolio_provider = portfolio_provider
        self.risk_computation = risk_computation
        
        # Initialize registry and portfolio graph
        self.risk_model_registry = RiskModelRegistry()
        self._portfolio_graph: Optional['PortfolioGraph'] = None
        self._current_risk_model: Optional[str] = None
        
        logger.info("Initialized RiskAnalysisService")
    
    def initialize(self) -> bool:
        """
        Initialize the service with default settings and run initial analysis.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Starting service initialization...")
            
            # Initialize risk models
            if not self._initialize_risk_models():
                logger.error("Failed to initialize risk models")
                return False
            
            # Build portfolio graph
            if not self._build_portfolio_graph():
                logger.error("Failed to build portfolio graph")
                return False
            
            # Set up risk computation
            if not self._setup_risk_computation():
                logger.error("Failed to setup risk computation")
                return False
            
            # Run initial analysis
            if not self._run_initial_analysis():
                logger.error("Failed to run initial analysis")
                return False
            
            logger.info("Service initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            return False
    
    def _initialize_risk_models(self) -> bool:
        """Initialize risk model registry with available models."""
        try:
            # Get available risk models from factor data
            available_models = self.factor_provider.get_available_risk_models()
            
            if not available_models:
                logger.warning("No risk models found in factor data")
                return False
            
            # Register models
            for model_code in available_models:
                factors = self.factor_provider.get_factor_names(model_code)
                date_range = self.factor_provider.get_date_range(model_code)
                
                metadata = {
                    "name": f"Risk Model {model_code}",
                    "description": f"Factor-based risk model with {len(factors)} factors",
                    "factors": factors,
                    "date_range": {
                        "start": date_range[0].isoformat(),
                        "end": date_range[1].isoformat()
                    },
                    "factor_count": len(factors)
                }
                
                self.risk_model_registry.register_model(model_code, metadata)
            
            # Set current model from config
            default_model = self.config_service.get_default_risk_model()
            if default_model in available_models:
                self.risk_model_registry.set_current_model(default_model)
                self._current_risk_model = default_model
            else:
                # Use first available model
                self._current_risk_model = available_models[0]
                self.risk_model_registry.set_current_model(self._current_risk_model)
            
            logger.info(f"Initialized {len(available_models)} risk models, "
                       f"current: {self._current_risk_model}")
            return True
            
        except Exception as e:
            logger.error(f"Risk model initialization failed: {e}")
            return False
    
    def _build_portfolio_graph(self) -> bool:
        """Get portfolio graph from YAML-based data provider."""
        try:
            # Get the pre-built PortfolioGraph from the portfolio provider
            self._portfolio_graph = self.portfolio_provider.get_portfolio_graph()
            
            if self._portfolio_graph is None:
                logger.error("Portfolio provider did not build a PortfolioGraph")
                return False
            
            # Validate the graph has components
            if not self._portfolio_graph.components:
                logger.error("PortfolioGraph has no components")
                return False
            
            # Get portfolio metadata for logging
            metadata = self.portfolio_provider.get_portfolio_metadata()
            portfolio_name = metadata.get('name', 'Unknown')
            
            logger.info(f"Retrieved PortfolioGraph for '{portfolio_name}' with "
                       f"{len(self._portfolio_graph.components)} components")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to get PortfolioGraph from provider: {e}")
            return False
    
    def _setup_risk_computation(self) -> bool:
        """Set up risk computation engine."""
        if not self._portfolio_graph:
            logger.error("Portfolio graph not available for risk computation setup")
            return False
        
        try:
            if self.risk_computation is None:
                self.risk_computation = RiskComputation(self._portfolio_graph)
            
            logger.info("Risk computation setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Risk computation setup failed: {e}")
            return False
    
    def _run_initial_analysis(self) -> bool:
        """Run initial risk analysis with current settings."""
        try:
            # Comprehensive validation before attempting analysis
            validation_result = self._validate_analysis_prerequisites()
            if not validation_result['valid']:
                logger.error(f"Analysis prerequisites not met: {validation_result['errors']}")
                return False
            
            if not self._current_risk_model:
                logger.error("No current risk model for initial analysis")
                return False
            
            # Get factor returns for current model
            logger.info(f"Loading factor returns for model: {self._current_risk_model}")
            factor_returns = self.factor_provider.get_factor_returns_wide(self._current_risk_model)
            
            if factor_returns.empty:
                logger.error("No factor returns data for initial analysis")
                return False
            
            logger.info(f"Loaded factor returns: {factor_returns.shape[0]} dates, {factor_returns.shape[1]} factors")
            
            # Run risk computation
            logger.info("Starting risk computation...")
            success = self.risk_computation.run_full_decomposition(factor_returns)
            
            if success:
                logger.info("Initial risk analysis completed successfully")
                
                # Validate that results were actually stored
                self._validate_risk_results()
                return True
            else:
                logger.error("Initial risk analysis failed")
                return False
                
        except Exception as e:
            logger.error(f"Initial analysis failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def switch_risk_model(self, model_code: str) -> bool:
        """
        Switch to a different risk model and recompute analysis.
        
        Args:
            model_code: Risk model code to switch to
            
        Returns:
            True if switch and recomputation successful
        """
        try:
            # Validate model exists
            available_models = self.risk_model_registry.list_models()
            model_codes = [m['code'] for m in available_models]
            
            if model_code not in model_codes:
                logger.error(f"Risk model not available: {model_code}")
                return False
            
            # Update registry
            if not self.risk_model_registry.set_current_model(model_code):
                logger.error(f"Failed to set current model: {model_code}")
                return False
            
            self._current_risk_model = model_code
            
            # Mark computation as stale
            if self.risk_computation:
                self.risk_computation.mark_stale()
            
            # Recompute with new model
            success = self.refresh_analysis(force=True)
            
            if success:
                logger.info(f"Successfully switched to risk model: {model_code}")
                return True
            else:
                logger.error(f"Failed to recompute after switching to: {model_code}")
                return False
                
        except Exception as e:
            logger.error(f"Risk model switch failed: {e}")
            return False
    
    def refresh_analysis(self, force: bool = False) -> bool:
        """
        Refresh risk analysis if needed or forced.
        
        Args:
            force: Force recomputation even if not stale
            
        Returns:
            True if refresh successful
        """
        try:
            if not self.risk_computation:
                logger.error("No risk computation engine available")
                return False
            
            if not force and not self.risk_computation.needs_recomputation():
                logger.info("Analysis is up to date, no refresh needed")
                return True
            
            if not self._current_risk_model:
                logger.error("No current risk model for refresh")
                return False
            
            # Get fresh factor returns
            factor_returns = self.factor_provider.get_factor_returns_wide(self._current_risk_model)
            
            if factor_returns.empty:
                logger.error("No factor returns data for refresh")
                return False
            
            # Run computation
            success = self.risk_computation.run_full_decomposition(factor_returns)
            
            if success:
                logger.info("Analysis refresh completed successfully")
                return True
            else:
                logger.error("Analysis refresh failed")
                return False
                
        except Exception as e:
            logger.error(f"Analysis refresh failed: {e}")
            return False
    
    def get_portfolio_graph(self) -> Optional['PortfolioGraph']:
        """
        Get the portfolio graph instance.
        
        Returns:
            PortfolioGraph instance or None if not initialized
        """
        return self._portfolio_graph
    
    def get_current_risk_model(self) -> str:
        """
        Get current risk model code.
        
        Returns:
            Current risk model code
        """
        return self._current_risk_model or ""
    
    def get_analysis_status(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis status.
        
        Returns:
            Dictionary with analysis status information
        """
        status = {
            "initialized": self._portfolio_graph is not None,
            "current_risk_model": self._current_risk_model,
            "portfolio_components": len(self._portfolio_graph.components) if self._portfolio_graph else 0,
            "risk_models_available": len(self.risk_model_registry.list_models()),
            "timestamp": datetime.now().isoformat()
        }
        
        if self.risk_computation:
            computation_status = {
                "computed": self.risk_computation.is_computed(),
                "stale": self.risk_computation.needs_recomputation(),
                "last_computation": self.risk_computation.get_computation_timestamp().isoformat() 
                                   if self.risk_computation.get_computation_timestamp() else None,
                "computation_stats": self.risk_computation.get_computation_stats()
            }
            status.update(computation_status)
        
        return status
    
    # Passthrough methods to computation layer
    def get_risk_results(self, component_id: str, lens: str) -> Dict[str, Any]:
        """
        Get risk results for a component and lens, formatted for UI.
        
        Args:
            component_id: Component identifier
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Dictionary with risk results formatted for UI
        """
        if not self.risk_computation:
            return {"error": "Risk computation not available"}
        
        risk_result = self.risk_computation.get_risk_result(component_id, lens)
        
        if not risk_result:
            return {"error": f"No risk results for {component_id}/{lens}"}
        
        # Format for UI consumption
        result_dict = {
            "component_id": component_id,
            "lens": lens,
            "total_risk": risk_result.total_risk,
            "factor_risk_contribution": risk_result.factor_risk_contribution,
            "specific_risk_contribution": risk_result.specific_risk_contribution,
            "cross_correlation_risk_contribution": risk_result.cross_correlation_risk_contribution,
            "factor_volatility": risk_result.factor_volatility,
            "specific_volatility": risk_result.specific_volatility,
            "cross_correlation_volatility": risk_result.cross_correlation_volatility,
            "factor_contributions": risk_result.factor_contributions,
            "asset_contributions": risk_result.asset_contributions,
            "factor_exposures": risk_result.factor_exposures,
            "portfolio_weights": risk_result.portfolio_weights,
            
            # Matrix data for heatmaps (previously missing)
            "weighted_betas": risk_result.weighted_betas or {},
            "asset_by_factor_contributions": risk_result.asset_by_factor_contributions or {},
            
            # Asset name mapping for visualization
            "asset_name_mapping": risk_result.asset_name_mapping or {}
        }
        
        # Log matrix data availability for debugging
        has_weighted_betas = risk_result.weighted_betas is not None and len(risk_result.weighted_betas) > 0
        has_asset_by_factor = risk_result.asset_by_factor_contributions is not None and len(risk_result.asset_by_factor_contributions) > 0
        
        logger.debug(f"Risk result for {component_id}/{lens}: weighted_betas={has_weighted_betas}, "
                    f"asset_by_factor_contributions={has_asset_by_factor}")
        
        if not has_weighted_betas:
            logger.warning(f"weighted_betas is empty for {component_id}/{lens}")
        if not has_asset_by_factor:
            logger.warning(f"asset_by_factor_contributions is empty for {component_id}/{lens}")
        
        return result_dict
    
    def get_all_components_risk(self, lens: str) -> Dict[str, Dict[str, Any]]:
        """
        Get risk results for all components for a specific lens.
        
        Args:
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Dictionary mapping component IDs to risk result dictionaries
        """
        if not self._portfolio_graph:
            return {}
        
        results = {}
        for component_id in self._portfolio_graph.components.keys():
            risk_data = self.get_risk_results(component_id, lens)
            if "error" not in risk_data:
                results[component_id] = risk_data
        
        return results
    
    def get_factor_contributions(self, component_id: str, lens: str) -> Dict[str, float]:
        """
        Get factor contributions for a component and lens.
        
        Args:
            component_id: Component identifier
            lens: Risk lens
            
        Returns:
            Dictionary mapping factor names to contribution values
        """
        risk_data = self.get_risk_results(component_id, lens)
        return risk_data.get("factor_contributions", {})
    
    def get_factor_exposures(self, component_id: str, lens: str) -> Dict[str, float]:
        """
        Get factor exposures for a component and lens.
        
        Args:
            component_id: Component identifier
            lens: Risk lens
            
        Returns:
            Dictionary mapping factor names to exposure values
        """
        risk_data = self.get_risk_results(component_id, lens)
        return risk_data.get("factor_exposures", {})
    
    def get_service_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive service summary.
        
        Returns:
            Dictionary with service summary
        """
        summary = {
            "service_status": "initialized" if self._portfolio_graph else "not_initialized",
            "analysis_status": self.get_analysis_status(),
            "risk_model_registry": self.risk_model_registry.get_registry_stats(),
            "configuration": self.config_service.get_config_summary(),
            "data_providers": {
                "factor_provider_available": self.factor_provider is not None,
                "portfolio_provider_available": self.portfolio_provider is not None
            }
        }
        
        if self.risk_computation:
            summary["computation_summary"] = self.risk_computation.get_computation_summary()
        
        return summary
    
    def _validate_analysis_prerequisites(self) -> Dict[str, Any]:
        """
        Validate that all prerequisites for risk analysis are met.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check portfolio graph
            if not self._portfolio_graph:
                validation['errors'].append("PortfolioGraph not initialized")
            else:
                if not self._portfolio_graph.components:
                    validation['errors'].append("PortfolioGraph has no components")
                
                if not self._portfolio_graph.root_id:
                    validation['errors'].append("PortfolioGraph root_id not set")
                elif self._portfolio_graph.root_id not in self._portfolio_graph.components:
                    validation['errors'].append(f"Root component '{self._portfolio_graph.root_id}' not found in graph components")
                    validation['errors'].append(f"Available components: {list(self._portfolio_graph.components.keys())}")
                
                if not self._portfolio_graph.metric_store:
                    validation['errors'].append("PortfolioGraph metric store not available")
            
            # Check risk computation
            if not self.risk_computation:
                validation['errors'].append("RiskComputation not initialized")
            
            # Check data providers
            if not self.factor_provider:
                validation['errors'].append("FactorDataProvider not available")
            
            if not self.portfolio_provider:
                validation['errors'].append("PortfolioDataProvider not available")
            
            # Check risk model
            if not self._current_risk_model:
                validation['errors'].append("No current risk model set")
            
            validation['valid'] = len(validation['errors']) == 0
            
        except Exception as e:
            validation['errors'].append(f"Error during validation: {e}")
            validation['valid'] = False
        
        return validation
    
    def _validate_risk_results(self) -> None:
        """
        Validate that risk results were properly generated and stored.
        """
        try:
            if not self.risk_computation:
                logger.warning("Cannot validate risk results: no risk computation available")
                return
            
            # Get a summary of what was stored
            metric_summary = self.risk_computation.get_metric_store_summary()
            
            # Check if the root component has risk results
            root_id = self._portfolio_graph.root_id if self._portfolio_graph else None
            
            if root_id and root_id in metric_summary.get("components_with_metrics", {}):
                root_metrics = metric_summary["components_with_metrics"][root_id]
                risk_results = root_metrics.get("risk_results", {})
                
                has_portfolio = risk_results.get("portfolio", False)
                has_benchmark = risk_results.get("benchmark", False) 
                has_active = risk_results.get("active", False)
                
                logger.info(f"Risk results validation for {root_id}: portfolio={has_portfolio}, benchmark={has_benchmark}, active={has_active}")
                
                if not has_active:
                    logger.warning(f"Missing active risk results for root component: {root_id}")
                
                if not (has_portfolio or has_benchmark):
                    logger.error(f"Missing both portfolio and benchmark risk results for root component: {root_id}")
            else:
                logger.error(f"No risk results found for root component: {root_id}")
                
        except Exception as e:
            logger.error(f"Error validating risk results: {e}")
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """
        Get detailed service status for comprehensive debugging.
        
        Returns:
            Dictionary with detailed status information
        """
        status = self.get_analysis_status()
        
        try:
            # Add validation results
            status['prerequisites'] = self._validate_analysis_prerequisites()
            
            # Add portfolio graph details
            if self._portfolio_graph:
                status['portfolio_graph'] = {
                    'root_id': self._portfolio_graph.root_id,
                    'components_count': len(self._portfolio_graph.components),
                    'components_list': list(self._portfolio_graph.components.keys()),
                    'metric_store_available': self._portfolio_graph.metric_store is not None
                }
            
            # Add risk computation details
            if self.risk_computation:
                status['risk_computation'] = self.risk_computation.get_metric_store_summary()
                
        except Exception as e:
            status['detailed_status_error'] = f"Error getting detailed status: {e}"
        
        return status