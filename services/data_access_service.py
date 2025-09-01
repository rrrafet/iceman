"""
Data Access Service - UI-friendly data access for portfolio risk analysis system.
Provides comprehensive data access methods optimized for UI consumption.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from services.risk_analysis_service import RiskAnalysisService

logger = logging.getLogger(__name__)


class DataAccessService:
    """
    Provides UI-friendly data access methods for the portfolio risk analysis system.
    
    Acts as a facade over the risk analysis service and data providers, offering
    methods specifically designed for UI consumption with appropriate formatting
    and error handling.
    """
    
    def __init__(self, risk_analysis_service: RiskAnalysisService):
        """
        Initialize data access service.
        
        Args:
            risk_analysis_service: Risk analysis service instance
        """
        self.risk_analysis_service = risk_analysis_service
        self.factor_provider = risk_analysis_service.factor_provider
        self.portfolio_provider = risk_analysis_service.portfolio_provider
        
        logger.info("Initialized DataAccessService")
    
    # Time series access methods
    def get_portfolio_returns(self, component_id: str) -> pd.Series:
        """
        Get portfolio returns time series for a component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Series with dates as index and portfolio returns as values
        """
        try:
            return self.portfolio_provider.get_component_returns(component_id, 'portfolio')
        except Exception as e:
            logger.error(f"Error getting portfolio returns for {component_id}: {e}")
            return pd.Series(dtype=float, name=f"{component_id}_portfolio")
    
    def get_benchmark_returns(self, component_id: str) -> pd.Series:
        """
        Get benchmark returns time series for a component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Series with dates as index and benchmark returns as values
        """
        try:
            return self.portfolio_provider.get_component_returns(component_id, 'benchmark')
        except Exception as e:
            logger.error(f"Error getting benchmark returns for {component_id}: {e}")
            return pd.Series(dtype=float, name=f"{component_id}_benchmark")
    
    def get_active_returns(self, component_id: str) -> pd.Series:
        """
        Get active returns (portfolio - benchmark) for a component.
        
        Args:
            component_id: Component identifier
            
        Returns:
            Series with dates as index and active returns as values
        """
        try:
            portfolio_returns = self.get_portfolio_returns(component_id)
            benchmark_returns = self.get_benchmark_returns(component_id)
            
            if portfolio_returns.empty or benchmark_returns.empty:
                return pd.Series(dtype=float, name=f"{component_id}_active")
            
            # Align series and compute active returns
            aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns)
            active_returns = aligned_portfolio - aligned_benchmark
            active_returns.name = f"{component_id}_active"
            
            return active_returns.dropna()
            
        except Exception as e:
            logger.error(f"Error computing active returns for {component_id}: {e}")
            return pd.Series(dtype=float, name=f"{component_id}_active")
    
    def get_all_component_ids(self) -> List[str]:
        """
        Get all available component IDs from the portfolio.
        
        Returns:
            List of component IDs
        """
        try:
            return self.portfolio_provider.get_all_component_ids()
        except Exception as e:
            logger.error(f"Error getting component IDs: {e}")
            return []
    
    def get_cumulative_returns(self, component_id: str, return_type: str) -> pd.Series:
        """
        Get cumulative returns for a component.
        
        Args:
            component_id: Component identifier
            return_type: Type of returns ('portfolio', 'benchmark', 'active')
            
        Returns:
            Series with cumulative returns
        """
        try:
            if return_type == 'portfolio':
                returns = self.get_portfolio_returns(component_id)
            elif return_type == 'benchmark':
                returns = self.get_benchmark_returns(component_id)
            elif return_type == 'active':
                returns = self.get_active_returns(component_id)
            else:
                logger.error(f"Invalid return_type: {return_type}")
                return pd.Series(dtype=float)
            
            if returns.empty:
                return pd.Series(dtype=float, name=f"{component_id}_cumulative_{return_type}")
            
            # Calculate cumulative returns: (1 + r).cumprod() - 1
            cumulative = (1 + returns).cumprod() - 1
            cumulative.name = f"{component_id}_cumulative_{return_type}"
            
            return cumulative
            
        except Exception as e:
            logger.error(f"Error computing cumulative returns for {component_id}: {e}")
            return pd.Series(dtype=float, name=f"{component_id}_cumulative_{return_type}")
    
    def get_returns_dataframe(self, component_ids: List[str], return_type: str) -> pd.DataFrame:
        """
        Get returns matrix for multiple components.
        
        Args:
            component_ids: List of component identifiers
            return_type: Type of returns ('portfolio', 'benchmark', 'active')
            
        Returns:
            DataFrame with dates as index and components as columns
        """
        try:
            if return_type in ['portfolio', 'benchmark']:
                return self.portfolio_provider.get_returns_matrix(return_type)
            elif return_type == 'active':
                # Compute active returns matrix
                portfolio_matrix = self.portfolio_provider.get_returns_matrix('portfolio')
                benchmark_matrix = self.portfolio_provider.get_returns_matrix('benchmark')
                
                if portfolio_matrix.empty or benchmark_matrix.empty:
                    return pd.DataFrame()
                
                # Filter to requested components
                common_components = list(set(component_ids) & set(portfolio_matrix.columns) & set(benchmark_matrix.columns))
                
                if not common_components:
                    return pd.DataFrame()
                
                portfolio_filtered = portfolio_matrix[common_components]
                benchmark_filtered = benchmark_matrix[common_components]
                
                # Compute active returns
                active_matrix = portfolio_filtered - benchmark_filtered
                return active_matrix.dropna()
            else:
                logger.error(f"Invalid return_type: {return_type}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting returns dataframe: {e}")
            return pd.DataFrame()
    
    def get_rolling_volatility(self, component_id: str, window: int, return_type: str) -> pd.Series:
        """
        Get rolling volatility for a component.
        
        Args:
            component_id: Component identifier
            window: Rolling window size in days
            return_type: Type of returns ('portfolio', 'benchmark', 'active')
            
        Returns:
            Series with rolling volatility
        """
        try:
            if return_type == 'portfolio':
                returns = self.get_portfolio_returns(component_id)
            elif return_type == 'benchmark':
                returns = self.get_benchmark_returns(component_id)
            elif return_type == 'active':
                returns = self.get_active_returns(component_id)
            else:
                logger.error(f"Invalid return_type: {return_type}")
                return pd.Series(dtype=float)
            
            if returns.empty or len(returns) < window:
                return pd.Series(dtype=float, name=f"{component_id}_volatility_{return_type}")
            
            # Calculate rolling volatility (annualized)
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            rolling_vol.name = f"{component_id}_volatility_{return_type}"
            
            return rolling_vol.dropna()
            
        except Exception as e:
            logger.error(f"Error computing rolling volatility for {component_id}: {e}")
            return pd.Series(dtype=float, name=f"{component_id}_volatility_{return_type}")
    
    # Risk metrics access methods
    def get_total_risk(self, component_id: str, lens: str) -> float:
        """
        Get total risk for a component and lens.
        
        Args:
            component_id: Component identifier
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Total risk value or 0.0 if not available
        """
        try:
            risk_data = self.risk_analysis_service.get_risk_results(component_id, lens)
            return risk_data.get("total_risk", 0.0)
        except Exception as e:
            logger.error(f"Error getting total risk for {component_id}/{lens}: {e}")
            return 0.0
    
    def get_factor_risk(self, component_id: str, lens: str) -> float:
        """
        Get factor risk for a component and lens.
        
        Args:
            component_id: Component identifier
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Factor risk value or 0.0 if not available
        """
        try:
            risk_data = self.risk_analysis_service.get_risk_results(component_id, lens)
            return risk_data.get("factor_risk", 0.0)
        except Exception as e:
            logger.error(f"Error getting factor risk for {component_id}/{lens}: {e}")
            return 0.0
    
    def get_specific_risk(self, component_id: str, lens: str) -> float:
        """
        Get specific risk for a component and lens.
        
        Args:
            component_id: Component identifier
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Specific risk value or 0.0 if not available
        """
        try:
            risk_data = self.risk_analysis_service.get_risk_results(component_id, lens)
            return risk_data.get("specific_risk", 0.0)
        except Exception as e:
            logger.error(f"Error getting specific risk for {component_id}/{lens}: {e}")
            return 0.0
    
    def get_risk_decomposition(self, component_id: str, lens: str) -> Dict[str, Any]:
        """
        Get comprehensive risk decomposition for a component and lens.
        
        Args:
            component_id: Component identifier
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Dictionary with complete risk decomposition data including:
            - Basic risk metrics (total_risk, factor_risk_contribution, etc.)
            - Factor-level data (factor_contributions, factor_exposures)
            - Asset-level data (asset_contributions, portfolio_weights)
        """
        try:
            risk_data = self.risk_analysis_service.get_risk_results(component_id, lens)
            
            # Check if risk_data contains an error or is None/empty
            if not risk_data or "error" in risk_data:
                error_msg = risk_data.get("error", "Unknown error") if risk_data else "No risk data returned"
                logger.error(f"Risk results unavailable for {component_id}/{lens}: {error_msg}")
                
                # Get detailed status for debugging
                debug_info = self.get_risk_computation_debug_info()
                logger.error(f"Debug info: {debug_info}")
                
                return self._get_empty_risk_decomposition_with_debug(component_id, lens, error_msg)
            
            # Extract matrix data with enhanced logging
            weighted_betas = risk_data.get("weighted_betas", {})
            asset_by_factor_contributions = risk_data.get("asset_by_factor_contributions", {})
            
            # Log detailed matrix data information
            logger.info(f"Matrix data for {component_id}/{lens}:")
            logger.info(f"  weighted_betas: type={type(weighted_betas).__name__}, "
                       f"empty={len(weighted_betas) == 0 if hasattr(weighted_betas, '__len__') else 'N/A'}")
            logger.info(f"  asset_by_factor_contributions: type={type(asset_by_factor_contributions).__name__}, "
                       f"empty={len(asset_by_factor_contributions) == 0 if hasattr(asset_by_factor_contributions, '__len__') else 'N/A'}")
            
            if weighted_betas:
                logger.info(f"  weighted_betas keys: {list(weighted_betas.keys())[:5]}...")  # First 5 keys
            if asset_by_factor_contributions:
                logger.info(f"  asset_by_factor_contributions keys: {list(asset_by_factor_contributions.keys())[:5]}...")  # First 5 keys
            
            return {
                # Basic risk metrics
                "factor_risk_contribution": risk_data.get("factor_risk_contribution", 0.0),
                "specific_risk_contribution": risk_data.get("specific_risk_contribution", 0.0),
                "cross_correlation_risk_contribution": risk_data.get("cross_correlation_risk_contribution", 0.0),
                "total_risk": risk_data.get("total_risk", 0.0),
                "factor_volatility": risk_data.get("factor_volatility", 0.0),
                "specific_volatility": risk_data.get("specific_volatility", 0.0),
                "cross_correlation_volatility": risk_data.get("cross_correlation_volatility", 0.0),
                # Factor-level data
                "factor_contributions": risk_data.get("factor_contributions", {}),
                "factor_exposures": risk_data.get("factor_exposures", {}),
                # Asset-level data
                "asset_contributions": risk_data.get("asset_contributions", {}),
                "portfolio_weights": risk_data.get("portfolio_weights", {}),
                # Matrix data (for heatmaps)
                "weighted_betas": weighted_betas,
                "asset_by_factor_contributions": asset_by_factor_contributions
            }
        except Exception as e:
            logger.error(f"Error getting risk decomposition for {component_id}/{lens}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return self._get_empty_risk_decomposition_with_debug(component_id, lens, str(e))

    def get_factor_exposure(self, component_id: str, lens: str) -> List[Tuple[str, float]]:
        """
        Get factor exposures for a component and lens.
        
        Args:
            component_id: Component identifier
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            List of tuples with (factor_name, exposure_value)
        """
        try:
            exposures = self.risk_analysis_service.get_factor_exposures(component_id, lens)
            return [(factor, exposure) for factor, exposure in exposures.items()]
        except Exception as e:
            logger.error(f"Error getting factor exposures for {component_id}/{lens}: {e}")
            return []
    
    def get_top_factor_contributors(self, component_id: str, lens: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N factor contributors for a component and lens.
        
        Args:
            component_id: Component identifier
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            n: Number of top contributors to return
            
        Returns:
            List of tuples with (factor_name, contribution_value) sorted by absolute contribution
        """
        try:
            contributions = self.risk_analysis_service.get_factor_contributions(component_id, lens)
            
            if not contributions:
                return []
            
            # Sort by absolute contribution value
            sorted_contributions = sorted(
                contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            return sorted_contributions[:n]
            
        except Exception as e:
            logger.error(f"Error getting top factor contributors for {component_id}/{lens}: {e}")
            return []
    
    def get_top_asset_contributors(self, component_id: str, lens: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N asset contributors for a component and lens.
        
        Args:
            component_id: Component identifier
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            n: Number of top contributors to return
            
        Returns:
            List of tuples with (asset_name, contribution_value) sorted by absolute contribution
        """
        try:
            risk_data = self.risk_analysis_service.get_risk_results(component_id, lens)
            asset_contributions = risk_data.get("asset_contributions", {})
            
            if not asset_contributions:
                return []
            
            # Sort by absolute contribution value
            sorted_contributions = sorted(
                asset_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            return sorted_contributions[:n]
            
        except Exception as e:
            logger.error(f"Error getting top asset contributors for {component_id}/{lens}: {e}")
            return []
    
    # Hierarchical analysis methods
    def get_risk_attribution_tree(self, root_id: str, lens: str) -> Dict[str, Any]:
        """
        Get hierarchical risk attribution tree.
        
        Args:
            root_id: Root component ID
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Dictionary with hierarchical risk breakdown
        """
        try:
            portfolio_graph = self.risk_analysis_service.get_portfolio_graph()
            if not portfolio_graph:
                return {}
            
            def build_tree(component_id: str) -> Dict[str, Any]:
                risk_data = self.risk_analysis_service.get_risk_results(component_id, lens)
                
                node_data = {
                    "component_id": component_id,
                    "total_risk": risk_data.get("total_risk", 0.0),
                    "factor_risk": risk_data.get("factor_risk", 0.0),
                    "specific_risk": risk_data.get("specific_risk", 0.0),
                    "children": []
                }
                
                # Add children if this is a node
                children = self.portfolio_provider.get_component_children(component_id)
                for child_id in children:
                    child_data = build_tree(child_id)
                    node_data["children"].append(child_data)
                
                return node_data
            
            return build_tree(root_id)
            
        except Exception as e:
            logger.error(f"Error building risk attribution tree for {root_id}/{lens}: {e}")
            return {}
    
    def get_component_children_risks(self, parent_id: str, lens: str) -> Dict[str, Dict[str, float]]:
        """
        Get risk metrics for all children of a parent component.
        
        Args:
            parent_id: Parent component ID
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Dictionary mapping child IDs to their risk metrics
        """
        try:
            children = self.portfolio_provider.get_component_children(parent_id)
            children_risks = {}
            
            for child_id in children:
                risk_decomp = self.get_risk_decomposition(child_id, lens)
                children_risks[child_id] = risk_decomp
            
            return children_risks
            
        except Exception as e:
            logger.error(f"Error getting children risks for {parent_id}/{lens}: {e}")
            return {}
    
    # Factor analysis methods
    def get_factor_returns_for_display(self, factor_names: List[str]) -> pd.DataFrame:
        """
        Get factor returns formatted for display.
        
        Args:
            factor_names: List of factor names to retrieve
            
        Returns:
            DataFrame with factor returns for display
        """
        try:
            current_model = self.risk_analysis_service.get_current_risk_model()
            if not current_model:
                return pd.DataFrame()
            
            factor_returns = self.factor_provider.get_factor_returns_wide(current_model)
            
            if factor_returns.empty:
                return pd.DataFrame()
            
            # Filter to requested factors
            available_factors = [f for f in factor_names if f in factor_returns.columns]
            
            if not available_factors:
                return pd.DataFrame()
            
            display_data = factor_returns[available_factors].copy()
            
            # Add summary statistics
            display_data = display_data.round(4)
            
            return display_data
            
        except Exception as e:
            logger.error(f"Error getting factor returns for display: {e}")
            return pd.DataFrame()
    
    # Weights and tilts methods
    def get_weights(self, component_id: str) -> Dict[str, float]:
        """
        Get weights for a component (portfolio, benchmark, active).
        
        Args:
            component_id: Component identifier
            
        Returns:
            Dictionary with portfolio, benchmark, and active weights
        """
        try:
            # Get latest weights (use most recent date available)
            portfolio_weights = self.portfolio_provider.get_component_weights(component_id, 'portfolio')
            benchmark_weights = self.portfolio_provider.get_component_weights(component_id, 'benchmark')
            
            weights = {}
            
            if not portfolio_weights.empty:
                weights['portfolio'] = float(portfolio_weights.iloc[-1])
            else:
                weights['portfolio'] = 0.0
            
            if not benchmark_weights.empty:
                weights['benchmark'] = float(benchmark_weights.iloc[-1])
            else:
                weights['benchmark'] = 0.0
            
            weights['active'] = weights['portfolio'] - weights['benchmark']
            
            return weights
            
        except Exception as e:
            logger.error(f"Error getting weights for {component_id}: {e}")
            return {"portfolio": 0.0, "benchmark": 0.0, "active": 0.0}
    
    # Summary statistics methods
    def get_risk_summary_stats(self, component_id: str, lens: str) -> Dict[str, Any]:
        """
        Get comprehensive risk summary statistics.
        
        Args:
            component_id: Component identifier
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Dictionary with risk summary statistics
        """
        try:
            risk_decomp = self.get_risk_decomposition(component_id, lens)
            top_factors = self.get_top_factor_contributors(component_id, lens, n=3)
            weights = self.get_weights(component_id)
            
            summary = {
                "component_id": component_id,
                "lens": lens,
                "risk_metrics": risk_decomp,
                "top_factor_contributors": top_factors,
                "weights": weights,
                "risk_ratios": {}
            }
            
            # Calculate risk ratios
            total_risk = risk_decomp.get("total_risk", 0.0)
            if total_risk > 0:
                summary["risk_ratios"]["factor_risk_ratio"] = risk_decomp.get("factor_risk", 0.0) / total_risk
                summary["risk_ratios"]["specific_risk_ratio"] = risk_decomp.get("specific_risk", 0.0) / total_risk
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting risk summary stats for {component_id}/{lens}: {e}")
            return {}
    
    def get_return_summary_stats(self, component_id: str, return_type: str) -> Dict[str, float]:
        """
        Get return summary statistics.
        
        Args:
            component_id: Component identifier
            return_type: Type of returns ('portfolio', 'benchmark', 'active')
            
        Returns:
            Dictionary with return summary statistics
        """
        try:
            if return_type == 'portfolio':
                returns = self.get_portfolio_returns(component_id)
            elif return_type == 'benchmark':
                returns = self.get_benchmark_returns(component_id)
            elif return_type == 'active':
                returns = self.get_active_returns(component_id)
            else:
                return {}
            
            if returns.empty:
                return {}
            
            # Calculate summary statistics
            summary = {
                "mean": float(returns.mean()),
                "std": float(returns.std()),
                "min": float(returns.min()),
                "max": float(returns.max()),
                "skew": float(returns.skew()),
                "kurtosis": float(returns.kurtosis()),
                "count": len(returns),
                "annualized_return": float(returns.mean() * 252),
                "annualized_volatility": float(returns.std() * np.sqrt(252))
            }
            
            # Sharpe ratio (if not active returns)
            if return_type != 'active' and summary["annualized_volatility"] > 0:
                summary["sharpe_ratio"] = summary["annualized_return"] / summary["annualized_volatility"]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting return summary stats for {component_id}/{return_type}: {e}")
            return {}
    
    def get_drawdown_analysis(self, component_id: str, return_type: str) -> Dict[str, Any]:
        """
        Get drawdown analysis for a component.
        
        Args:
            component_id: Component identifier
            return_type: Type of returns ('portfolio', 'benchmark', 'active')
            
        Returns:
            Dictionary with drawdown analysis
        """
        try:
            cumulative_returns = self.get_cumulative_returns(component_id, return_type)
            
            if cumulative_returns.empty:
                return {}
            
            # Calculate running maximum and drawdowns
            running_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns - running_max
            
            analysis = {
                "max_drawdown": float(drawdowns.min()),
                "current_drawdown": float(drawdowns.iloc[-1]),
                "max_drawdown_date": str(drawdowns.idxmin()),
                "drawdown_duration_days": 0,  # Would need more complex calculation
                "recovery_periods": []  # Would need more complex calculation
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting drawdown analysis for {component_id}/{return_type}: {e}")
            return {}
    
    def get_available_factors(self) -> List[str]:
        """
        Get list of available factor names.
        
        Returns:
            List of factor names from the factor provider
        """
        try:
            # Get default risk model to get factor names
            risk_models = self.factor_provider.get_available_risk_models()
            if risk_models:
                return self.factor_provider.get_factor_names(risk_models[0])
            return []
        except Exception as e:
            logger.error(f"Error getting available factors: {e}")
            return []
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status for debugging.
        
        Returns:
            Dictionary with service status information
        """
        try:
            status = {
                "data_access_service": "active",
                "risk_analysis_service_status": self.risk_analysis_service.get_analysis_status(),
                "data_providers": {
                    "factor_provider_status": "active" if self.factor_provider else "inactive",
                    "portfolio_provider_status": "active" if self.portfolio_provider else "inactive"
                },
                "available_methods": {
                    "time_series_methods": [
                        "get_portfolio_returns", "get_benchmark_returns", "get_active_returns",
                        "get_cumulative_returns", "get_returns_dataframe", "get_rolling_volatility"
                    ],
                    "risk_methods": [
                        "get_total_risk", "get_factor_risk", "get_specific_risk",
                        "get_risk_decomposition", "get_factor_exposure", "get_top_factor_contributors"
                    ],
                    "analysis_methods": [
                        "get_risk_attribution_tree", "get_component_children_risks",
                        "get_risk_summary_stats", "get_return_summary_stats"
                    ]
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {"error": str(e)}
    
    def get_risk_computation_debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about risk computation status.
        
        Returns:
            Dictionary with detailed debug information
        """
        try:
            debug_info = {
                "service_available": self.risk_analysis_service is not None,
                "detailed_status": {}
            }
            
            if self.risk_analysis_service:
                # Get detailed status from risk analysis service
                if hasattr(self.risk_analysis_service, 'get_detailed_status'):
                    debug_info["detailed_status"] = self.risk_analysis_service.get_detailed_status()
                else:
                    debug_info["detailed_status"] = self.risk_analysis_service.get_analysis_status()
            
            return debug_info
            
        except Exception as e:
            return {"debug_error": f"Error getting debug info: {e}"}
    
    def _get_empty_risk_decomposition_with_debug(self, component_id: str, lens: str, error_msg: str) -> Dict[str, Any]:
        """
        Return empty risk decomposition with debug information.
        
        Args:
            component_id: Component identifier
            lens: Risk lens
            error_msg: Error message
            
        Returns:
            Empty risk decomposition with debug info
        """
        return {
            "factor_risk_contribution": 0.0, 
            "specific_risk_contribution": 0.0, 
            "cross_correlation_risk_contribution": 0.0, 
            "total_risk": 0.0,
            "factor_volatility": 0.0,
            "specific_volatility": 0.0,
            "cross_correlation_volatility": 0.0,
            "factor_contributions": {},
            "factor_exposures": {},
            "asset_contributions": {},
            "portfolio_weights": {},
            "weighted_betas": {},
            "asset_by_factor_contributions": {},
            # Debug information
            "_debug_info": {
                "error": error_msg,
                "component_id": component_id,
                "lens": lens,
                "timestamp": pd.Timestamp.now().isoformat()
            }
        }
    
    def validate_risk_computation_setup(self) -> Dict[str, Any]:
        """
        Validate that risk computation is properly set up.
        
        Returns:
            Dictionary with validation results
        """
        try:
            validation = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "components": {}
            }
            
            # Check if risk analysis service is available
            if not self.risk_analysis_service:
                validation["errors"].append("RiskAnalysisService not available")
                validation["valid"] = False
                return validation
            
            # Get detailed status
            if hasattr(self.risk_analysis_service, 'get_detailed_status'):
                status = self.risk_analysis_service.get_detailed_status()
                
                # Check prerequisites
                prereqs = status.get('prerequisites', {})
                if not prereqs.get('valid', False):
                    validation["errors"].extend(prereqs.get('errors', []))
                    validation["warnings"].extend(prereqs.get('warnings', []))
                
                # Check portfolio graph
                portfolio_graph = status.get('portfolio_graph', {})
                if not portfolio_graph:
                    validation["errors"].append("PortfolioGraph status not available")
                else:
                    validation["components"]["portfolio_graph"] = portfolio_graph
                
                # Check risk computation
                risk_computation = status.get('risk_computation', {})
                validation["components"]["risk_computation"] = risk_computation
            
            validation["valid"] = len(validation["errors"]) == 0
            return validation
            
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Error during validation: {e}"],
                "warnings": [],
                "components": {}
            }
    
    def debug_matrix_data_availability(self, component_id: str = "TOTAL", lens: str = "active") -> Dict[str, Any]:
        """
        Debug method to investigate matrix data availability in the risk calculation pipeline.
        
        Args:
            component_id: Component to investigate (default: "TOTAL")
            lens: Risk lens to investigate (default: "active")
            
        Returns:
            Detailed debug information about matrix data
        """
        debug_info = {
            "component_id": component_id,
            "lens": lens,
            "timestamp": pd.Timestamp.now().isoformat(),
            "pipeline_stages": {}
        }
        
        try:
            # Stage 1: Check RiskAnalysisService
            if not self.risk_analysis_service:
                debug_info["pipeline_stages"]["risk_analysis_service"] = "NOT_AVAILABLE"
                return debug_info
            
            # Stage 2: Get raw risk result from computation
            risk_computation = getattr(self.risk_analysis_service, 'risk_computation', None)
            if risk_computation:
                raw_risk_result = risk_computation.get_risk_result(component_id, lens)
                
                debug_info["pipeline_stages"]["raw_risk_result"] = {
                    "available": raw_risk_result is not None,
                    "type": type(raw_risk_result).__name__ if raw_risk_result else None
                }
                
                if raw_risk_result:
                    # Check if raw result has matrix data
                    weighted_betas_raw = getattr(raw_risk_result, 'weighted_betas', None)
                    asset_by_factor_raw = getattr(raw_risk_result, 'asset_by_factor_contributions', None)
                    
                    debug_info["pipeline_stages"]["raw_risk_result"].update({
                        "weighted_betas": {
                            "exists": weighted_betas_raw is not None,
                            "type": type(weighted_betas_raw).__name__ if weighted_betas_raw is not None else None,
                            "empty": len(weighted_betas_raw) == 0 if weighted_betas_raw and hasattr(weighted_betas_raw, '__len__') else "N/A"
                        },
                        "asset_by_factor_contributions": {
                            "exists": asset_by_factor_raw is not None,
                            "type": type(asset_by_factor_raw).__name__ if asset_by_factor_raw is not None else None,
                            "empty": len(asset_by_factor_raw) == 0 if asset_by_factor_raw and hasattr(asset_by_factor_raw, '__len__') else "N/A"
                        }
                    })
            else:
                debug_info["pipeline_stages"]["raw_risk_result"] = "RISK_COMPUTATION_NOT_AVAILABLE"
            
            # Stage 3: Check formatted risk result from service
            formatted_risk_result = self.risk_analysis_service.get_risk_results(component_id, lens)
            debug_info["pipeline_stages"]["formatted_risk_result"] = {
                "available": formatted_risk_result is not None and "error" not in formatted_risk_result,
                "has_weighted_betas": "weighted_betas" in formatted_risk_result if formatted_risk_result else False,
                "has_asset_by_factor": "asset_by_factor_contributions" in formatted_risk_result if formatted_risk_result else False
            }
            
            if formatted_risk_result and "error" not in formatted_risk_result:
                weighted_betas_formatted = formatted_risk_result.get("weighted_betas", {})
                asset_by_factor_formatted = formatted_risk_result.get("asset_by_factor_contributions", {})
                
                debug_info["pipeline_stages"]["formatted_risk_result"].update({
                    "weighted_betas_empty": len(weighted_betas_formatted) == 0 if hasattr(weighted_betas_formatted, '__len__') else True,
                    "asset_by_factor_empty": len(asset_by_factor_formatted) == 0 if hasattr(asset_by_factor_formatted, '__len__') else True,
                    "weighted_betas_sample": list(weighted_betas_formatted.keys())[:3] if weighted_betas_formatted else [],
                    "asset_by_factor_sample": list(asset_by_factor_formatted.keys())[:3] if asset_by_factor_formatted else []
                })
            
            # Stage 4: Check final DataAccessService result
            final_result = self.get_risk_decomposition(component_id, lens)
            debug_info["pipeline_stages"]["final_result"] = {
                "available": final_result is not None,
                "has_debug_info": "_debug_info" in final_result if final_result else False,
                "weighted_betas_empty": len(final_result.get("weighted_betas", {})) == 0 if final_result else True,
                "asset_by_factor_empty": len(final_result.get("asset_by_factor_contributions", {})) == 0 if final_result else True
            }
            
        except Exception as e:
            debug_info["error"] = f"Debug method failed: {e}"
            import traceback
            debug_info["traceback"] = traceback.format_exc()
        
        return debug_info