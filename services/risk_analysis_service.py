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
            if not self._current_risk_model:
                logger.error("No current risk model for initial analysis")
                return False
            
            # Get factor returns for current model
            factor_returns = self.factor_provider.get_factor_returns_wide(self._current_risk_model)
            
            if factor_returns.empty:
                logger.error("No factor returns data for initial analysis")
                return False
            
            # Run risk computation
            success = self.risk_computation.run_full_decomposition(factor_returns)
            
            if success:
                logger.info("Initial risk analysis completed successfully")
                return True
            else:
                logger.error("Initial risk analysis failed")
                return False
                
        except Exception as e:
            logger.error(f"Initial analysis failed: {e}")
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
        return {
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
            "portfolio_weights": risk_result.portfolio_weights
        }
    
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