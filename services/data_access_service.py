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


class FrequencyManager:
    """Manages data frequency and resampling state."""
    
    SUPPORTED_FREQUENCIES = ["D", "B", "W-FRI", "M"]
    NATIVE_FREQUENCIES = ["D", "B"]
    
    def __init__(self, native_frequency: str = "B"):
        """
        Initialize frequency manager.
        
        Args:
            native_frequency: The native frequency of the data (default: "B" for business daily)
        """
        self.native_frequency = native_frequency
        self.current_frequency = native_frequency
        self.is_resampled = False
    
    def set_frequency(self, new_frequency: str) -> bool:
        """
        Set new frequency. Returns True if frequency changed.
        
        Args:
            new_frequency: New frequency to set
            
        Returns:
            True if frequency changed, False otherwise
            
        Raises:
            ValueError: If frequency is not supported
        """
        if new_frequency not in self.SUPPORTED_FREQUENCIES:
            raise ValueError(f"Unsupported frequency: {new_frequency}. Supported: {self.SUPPORTED_FREQUENCIES}")
        
        if new_frequency != self.current_frequency:
            old_frequency = self.current_frequency
            self.current_frequency = new_frequency
            self.is_resampled = new_frequency not in self.NATIVE_FREQUENCIES
            logger.info(f"Frequency changed from {old_frequency} to {new_frequency} (resampled: {self.is_resampled})")
            return True
        return False
    
    def get_current_frequency(self) -> str:
        """Get current frequency."""
        return self.current_frequency
    
    def is_native_frequency(self) -> bool:
        """Check if current frequency is native (no resampling needed)."""
        return not self.is_resampled


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
        
        # Initialize frequency manager
        self.frequency_manager = FrequencyManager()
        
        logger.info("Initialized DataAccessService with frequency manager")
    
    # Frequency management methods
    def set_frequency(self, frequency: str) -> bool:
        """
        Set data frequency and trigger re-initialization if needed.
        
        Args:
            frequency: New frequency to set ("D", "B", "W-FRI", "M")
            
        Returns:
            True if frequency changed and system was re-initialized, False otherwise
            
        Raises:
            ValueError: If frequency is not supported
        """
        try:
            frequency_changed = self.frequency_manager.set_frequency(frequency)
            
            if frequency_changed:
                logger.info(f"Frequency changed to {frequency}, triggering system re-initialization")
                self._trigger_system_reinitialization()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error setting frequency to {frequency}: {e}")
            raise
    
    def get_current_frequency(self) -> str:
        """Get current data frequency."""
        return self.frequency_manager.get_current_frequency()
    
    def get_frequency_status(self) -> Dict[str, Any]:
        """Get detailed frequency status for debugging."""
        return {
            "current_frequency": self.frequency_manager.current_frequency,
            "native_frequency": self.frequency_manager.native_frequency,
            "is_resampled": self.frequency_manager.is_resampled,
            "supported_frequencies": self.frequency_manager.SUPPORTED_FREQUENCIES
        }
    
    def is_native_frequency(self) -> bool:
        """Check if current frequency is native (no resampling)."""
        return self.frequency_manager.is_native_frequency()
    
    def _trigger_system_reinitialization(self):
        """Trigger full system re-initialization after frequency change."""
        try:
            logger.info("Starting system re-initialization for frequency change")
            
            # Clear portfolio graph and risk computation caches
            if hasattr(self.risk_analysis_service, '_portfolio_graph'):
                self.risk_analysis_service._portfolio_graph = None
                logger.info("Cleared portfolio graph cache")
            
            if hasattr(self.risk_analysis_service, 'risk_computation') and self.risk_analysis_service.risk_computation:
                if hasattr(self.risk_analysis_service.risk_computation, 'clear_cache'):
                    self.risk_analysis_service.risk_computation.clear_cache()
                    logger.info("Cleared risk computation cache")
            
            # Re-initialize the entire system
            logger.info("Re-initializing risk analysis service")
            success = self.risk_analysis_service.initialize()
            
            if not success:
                logger.error("Failed to re-initialize system after frequency change")
                raise RuntimeError("System re-initialization failed")
            else:
                logger.info("System re-initialization completed successfully")
                
        except Exception as e:
            logger.error(f"Error during system re-initialization: {e}")
            raise RuntimeError(f"Failed to re-initialize system: {e}")
    
    # Resampling helper methods
    def _resample_returns_series(self, series: pd.Series) -> pd.Series:
        """
        Resample returns series using compound return calculation.
        
        Args:
            series: Returns series to resample
            
        Returns:
            Resampled series or original series if no resampling needed
        """
        if not self.frequency_manager.is_resampled or series.empty:
            return series
        
        freq = self.frequency_manager.current_frequency
        try:
            # Use compound return calculation: (1+r).prod() - 1
            resampled = series.resample(freq).apply(lambda x: (1 + x).prod() - 1)
            resampled.name = series.name
            logger.debug(f"Resampled series {series.name} from {len(series)} to {len(resampled)} observations at {freq} frequency")
            return resampled.dropna()
            
        except Exception as e:
            logger.error(f"Error resampling series {series.name} to {freq}: {e}")
            return series
    
    def _resample_returns_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample returns DataFrame using compound return calculation.
        
        Args:
            df: Returns DataFrame to resample
            
        Returns:
            Resampled DataFrame or original DataFrame if no resampling needed
        """
        if not self.frequency_manager.is_resampled or df.empty:
            return df
        
        freq = self.frequency_manager.current_frequency
        try:
            # Apply compound return calculation to each column
            resampled = df.resample(freq).apply(lambda x: (1 + x).prod() - 1)
            logger.debug(f"Resampled DataFrame from {len(df)} to {len(resampled)} observations at {freq} frequency")
            return resampled.dropna()
            
        except Exception as e:
            logger.error(f"Error resampling DataFrame to {freq}: {e}")
            return df
    
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
            returns = self.portfolio_provider.get_component_returns(component_id, 'portfolio')
            return self._resample_returns_series(returns)
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
            returns = self.portfolio_provider.get_component_returns(component_id, 'benchmark')
            return self._resample_returns_series(returns)
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
            # Get resampled portfolio and benchmark returns separately
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
                matrix = self.portfolio_provider.get_returns_matrix(return_type)
                return self._resample_returns_dataframe(matrix)
            elif return_type == 'active':
                # Compute active returns matrix
                portfolio_matrix = self.portfolio_provider.get_returns_matrix('portfolio')
                benchmark_matrix = self.portfolio_provider.get_returns_matrix('benchmark')
                
                if portfolio_matrix.empty or benchmark_matrix.empty:
                    return pd.DataFrame()
                
                # Apply resampling to both matrices before computing active returns
                portfolio_resampled = self._resample_returns_dataframe(portfolio_matrix)
                benchmark_resampled = self._resample_returns_dataframe(benchmark_matrix)
                
                # Filter to requested components
                common_components = list(set(component_ids) & set(portfolio_resampled.columns) & set(benchmark_resampled.columns))
                
                if not common_components:
                    return pd.DataFrame()
                
                portfolio_filtered = portfolio_resampled[common_components]
                benchmark_filtered = benchmark_resampled[common_components]
                
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
            # Adjust annualization factor based on frequency
            freq_annualization = {
                "D": 252,  # Daily
                "B": 252,  # Business daily
                "W-FRI": 52,  # Weekly
                "M": 12   # Monthly
            }
            annualization_factor = freq_annualization.get(self.frequency_manager.current_frequency, 252)
            
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(annualization_factor)
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
    
    def get_allocation_selection_decomposition(self, component_id: str, lens: str) -> Dict[str, Any]:
        """
        Get comprehensive risk decomposition data specifically for allocation-selection analysis.
        
        This method returns ALL necessary fields from RiskResult that are needed for the
        12-column Brinson-style allocation-selection table, including fields that are
        missing from the general-purpose get_risk_decomposition method.
        
        Args:
            component_id: Component identifier
            lens: Risk lens ('portfolio', 'benchmark', 'active')
            
        Returns:
            Dictionary with comprehensive risk decomposition data including:
            - All basic risk metrics and weights
            - Asset-level marginal contributions and factor/specific breakdowns
            - Active risk allocation and selection components (when applicable)
            - Metadata (analysis_type, annualized, frequency)
        """
        try:
            if not self.risk_analysis_service.risk_computation:
                return {"error": "Risk computation not available"}
            
            # Get the raw RiskResult object directly from computation
            risk_result = self.risk_analysis_service.risk_computation.get_risk_result(component_id, lens)
            
            if not risk_result:
                return {"error": f"No risk results for {component_id}/{lens}"}
            
            # Extract all necessary fields for allocation-selection analysis
            result_dict = {
                # Basic identifiers and metadata
                "component_id": component_id,
                "lens": lens,
                "analysis_type": risk_result.analysis_type,
                "annualized": risk_result.annualized,
                "frequency": risk_result.frequency,
                
                # Core risk metrics
                "total_risk": risk_result.total_risk,
                "factor_risk_contribution": risk_result.factor_risk_contribution,
                "specific_risk_contribution": risk_result.specific_risk_contribution,
                "cross_correlation_risk_contribution": getattr(risk_result, 'cross_correlation_risk_contribution', 0.0),
                
                # Weights (all three types)
                "portfolio_weights": risk_result.portfolio_weights or {},
                "benchmark_weights": risk_result.benchmark_weights or {},
                "active_weights": risk_result.active_weights or {},
                
                # Asset-level contributions (MISSING from get_risk_decomposition)
                "marginal_contributions": risk_result.marginal_contributions or {},
                "asset_contributions": risk_result.asset_contributions or {},
                "asset_factor_contributions": risk_result.asset_factor_contributions or {},
                "asset_specific_contributions": risk_result.asset_specific_contributions or {},
                
                # Active risk allocation and selection breakdowns (MISSING from get_risk_decomposition)
                "asset_allocation_factor": risk_result.asset_allocation_factor or {},
                "asset_allocation_specific": risk_result.asset_allocation_specific or {},
                "asset_selection_factor": risk_result.asset_selection_factor or {},
                "asset_selection_specific": risk_result.asset_selection_specific or {},
                "asset_cross_correlation": risk_result.asset_cross_correlation or {},
                
                # Factor-level data
                "factor_contributions": risk_result.factor_contributions or {},
                "factor_exposures": risk_result.factor_exposures or {},
                
                # Asset name mapping for display
                "asset_name_mapping": risk_result.asset_name_mapping or {},
                
                # Matrix data for heatmaps
                "weighted_betas": risk_result.weighted_betas or {},
                "asset_by_factor_contributions": risk_result.asset_by_factor_contributions or {},
                
                # Validation
                "validation_passed": risk_result.validation_passed,
                "validation_message": risk_result.validation_message
            }
            
            logger.info(f"Retrieved comprehensive allocation-selection data for {component_id}/{lens}")
            logger.debug(f"Analysis type: {risk_result.analysis_type}, "
                        f"Assets: {len(risk_result.asset_contributions)}, "
                        f"Has allocation data: {risk_result.asset_allocation_factor is not None}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Error getting allocation-selection decomposition for {component_id}/{lens}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return error response
            return {
                "error": f"Failed to get allocation-selection data: {str(e)}",
                "component_id": component_id,
                "lens": lens
            }

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
            
            # Calculate summary statistics with frequency-adjusted annualization
            freq_annualization = {
                "D": 252,  # Daily
                "B": 252,  # Business daily
                "W-FRI": 52,  # Weekly
                "M": 12   # Monthly
            }
            annualization_factor = freq_annualization.get(self.frequency_manager.current_frequency, 252)
            
            summary = {
                "mean": float(returns.mean()),
                "std": float(returns.std()),
                "min": float(returns.min()),
                "max": float(returns.max()),
                "skew": float(returns.skew()),
                "kurtosis": float(returns.kurtosis()),
                "count": len(returns),
                "annualized_return": float(returns.mean() * annualization_factor),
                "annualized_volatility": float(returns.std() * np.sqrt(annualization_factor))
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
    
    def get_descendant_returns_data(self, component_id: str, lens: str = "portfolio") -> pd.DataFrame:
        """
        Get returns data for all descendant components of a given component.
        
        Args:
            component_id: Component ID to get descendant returns for
            lens: Type of returns ('portfolio', 'benchmark', 'active')
            
        Returns:
            DataFrame with dates as index and component returns as columns
        """
        try:
            descendant_ids = self.get_descendant_leaf_ids(component_id)
            
            if not descendant_ids:
                return pd.DataFrame()
            
            returns_data = {}
            
            for desc_id in descendant_ids:
                if lens == "portfolio":
                    returns = self.get_portfolio_returns(desc_id)
                elif lens == "benchmark":
                    returns = self.get_benchmark_returns(desc_id)
                elif lens == "active":
                    returns = self.get_active_returns(desc_id)
                else:
                    continue
                    
                if not returns.empty:
                    returns_data[desc_id] = returns
            
            if not returns_data:
                return pd.DataFrame()
            
            # Combine all returns into a single DataFrame
            combined_df = pd.DataFrame(returns_data)
            return combined_df.dropna(how='all')
            
        except Exception as e:
            logger.error(f"Error getting descendant returns data: {e}")
            return pd.DataFrame()
    
    def get_factor_list(self) -> List[str]:
        """
        Get list of available factor names.
        
        Returns:
            List of factor names
        """
        try:
            current_model = self.risk_analysis_service.get_current_risk_model()
            if not current_model:
                return []
            
            factor_returns = self.factor_provider.get_factor_returns_wide(current_model)
            if factor_returns.empty:
                return []
                
            return list(factor_returns.columns)
            
        except Exception as e:
            logger.error(f"Error getting factor list: {e}")
            return []
    
    def get_factor_returns_data(self, factor_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get factor returns time series data.
        
        Args:
            factor_names: List of factor names to retrieve. If None, gets all factors.
            
        Returns:
            DataFrame with dates as index and factor returns as columns
        """
        try:
            current_model = self.risk_analysis_service.get_current_risk_model()
            if not current_model:
                return pd.DataFrame()
            
            factor_returns = self.factor_provider.get_factor_returns_wide(current_model)
            
            if factor_returns.empty:
                return pd.DataFrame()
            
            # Apply resampling to factor returns
            factor_returns_resampled = self._resample_returns_dataframe(factor_returns)
            
            if factor_names:
                # Filter to requested factors
                available_factors = [f for f in factor_names if f in factor_returns_resampled.columns]
                if available_factors:
                    return factor_returns_resampled[available_factors].copy()
                else:
                    return pd.DataFrame()
            else:
                return factor_returns_resampled.copy()
            
        except Exception as e:
            logger.error(f"Error getting factor returns data: {e}")
            return pd.DataFrame()

    def get_descendant_leaf_ids(self, component_id: str) -> List[str]:
        """
        Get all descendant leaf component IDs for a given component.
        
        Args:
            component_id: Component ID to get descendants for
            
        Returns:
            List of leaf component IDs that are descendants of the given component
        """
        try:
            portfolio_graph = self.risk_analysis_service.get_portfolio_graph()
            if not portfolio_graph:
                logger.warning(f"Portfolio graph not available for descendant search")
                return []
            
            def _collect_leaves(current_id: str) -> List[str]:
                """Recursively collect leaf nodes."""
                children = self.portfolio_provider.get_component_children(current_id)
                
                if not children:
                    # This is a leaf node
                    return [current_id]
                
                # Collect leaves from all children
                all_leaves = []
                for child_id in children:
                    all_leaves.extend(_collect_leaves(child_id))
                
                return all_leaves
            
            leaf_ids = _collect_leaves(component_id)
            logger.info(f"Found {len(leaf_ids)} descendant leaves for {component_id}")
            return sorted(leaf_ids)
            
        except Exception as e:
            logger.error(f"Error getting descendant leaf IDs for {component_id}: {e}")
            return []
    
    def riskresult_to_brinson_table(self, risk_result: Dict[str, Any], leaf_ids: List[str]) -> pd.DataFrame:
        """
        Transform RiskResult data into Brinson-style allocation-selection table.
        
        Creates exactly 12 columns as specified in allocation-selection requirements:
        1. asset_name, 2. portfolio_weight, 3. benchmark_weight, 4. asset marginal,
        5. asset contribution to risk, 6. asset contribution to factor risk,
        7. asset contribution to specific risk, 8. asset contribution to allocation factor risk,
        9. asset contribution to allocation specific risk, 10. asset contribution to selection factor risk,
        11. asset contribution to selection specific risk, 12. component_id
        
        Args:
            risk_result: Risk decomposition data from get_risk_decomposition
            leaf_ids: List of leaf component IDs to include in table
            
        Returns:
            DataFrame with exactly 12 columns, one row per leaf asset
        """
        try:
            if not risk_result or not leaf_ids:
                # Return empty DataFrame with correct column structure
                empty_df = pd.DataFrame(columns=[
                    'asset_name', 'portfolio_weight', 'benchmark_weight', 'asset marginal',
                    'asset contribution to risk', 'asset contribution to factor risk',
                    'asset contribution to specific risk', 'asset contribution to allocation factor risk',
                    'asset contribution to allocation specific risk', 'asset contribution to selection factor risk',
                    'asset contribution to selection specific risk', 'component_id'
                ])
                return empty_df
            
            # Extract data from risk result
            analysis_type = risk_result.get('analysis_type', 'portfolio')
            asset_name_mapping = risk_result.get('asset_name_mapping', {})
            portfolio_weights = risk_result.get('portfolio_weights', {})
            benchmark_weights = risk_result.get('benchmark_weights', {})
            marginal_contributions = risk_result.get('marginal_contributions', {})
            asset_contributions = risk_result.get('asset_contributions', {})
            
            # Portfolio risk decomposition
            asset_factor_contributions = risk_result.get('asset_factor_contributions', {})
            asset_specific_contributions = risk_result.get('asset_specific_contributions', {})
            
            # Active risk decomposition (Brinson-style)
            asset_allocation_factor = risk_result.get('asset_allocation_factor', {})
            asset_allocation_specific = risk_result.get('asset_allocation_specific', {})
            asset_selection_factor = risk_result.get('asset_selection_factor', {})
            asset_selection_specific = risk_result.get('asset_selection_specific', {})
            
            # Build table data
            table_data = []
            
            for component_id in leaf_ids:
                # Basic identifiers
                asset_name = asset_name_mapping.get(component_id, component_id) if asset_name_mapping else component_id
                
                # Weights
                portfolio_weight = portfolio_weights.get(component_id, np.nan)
                benchmark_weight = benchmark_weights.get(component_id, 0.0) if benchmark_weights else 0.0
                
                # Core risk metrics
                asset_marginal = marginal_contributions.get(component_id, np.nan)
                asset_contrib_to_risk = asset_contributions.get(component_id, np.nan)
                
                # Factor and specific risk contributions
                if analysis_type == "active":
                    # Active risk: combine allocation + selection for total factor/specific
                    alloc_factor = asset_allocation_factor.get(component_id, 0.0) if asset_allocation_factor else 0.0
                    alloc_specific = asset_allocation_specific.get(component_id, 0.0) if asset_allocation_specific else 0.0
                    select_factor = asset_selection_factor.get(component_id, 0.0) if asset_selection_factor else 0.0
                    select_specific = asset_selection_specific.get(component_id, 0.0) if asset_selection_specific else 0.0
                    
                    asset_contrib_to_factor = alloc_factor + select_factor
                    asset_contrib_to_specific = alloc_specific + select_specific
                    
                    # Allocation/Selection components
                    asset_contrib_alloc_factor = alloc_factor
                    asset_contrib_alloc_specific = alloc_specific
                    asset_contrib_select_factor = select_factor
                    asset_contrib_select_specific = select_specific
                    
                else:
                    # Portfolio risk: use asset_*_contributions, zero allocation/selection
                    asset_contrib_to_factor = asset_factor_contributions.get(component_id, np.nan)
                    asset_contrib_to_specific = asset_specific_contributions.get(component_id, np.nan)
                    
                    # No allocation/selection for portfolio lens
                    asset_contrib_alloc_factor = 0.0
                    asset_contrib_alloc_specific = 0.0
                    asset_contrib_select_factor = 0.0
                    asset_contrib_select_specific = 0.0
                
                # Build row
                row = {
                    'asset_name': asset_name,
                    'portfolio_weight': portfolio_weight,
                    'benchmark_weight': benchmark_weight,
                    'asset marginal': asset_marginal,
                    'asset contribution to risk': asset_contrib_to_risk,
                    'asset contribution to factor risk': asset_contrib_to_factor,
                    'asset contribution to specific risk': asset_contrib_to_specific,
                    'asset contribution to allocation factor risk': asset_contrib_alloc_factor,
                    'asset contribution to allocation specific risk': asset_contrib_alloc_specific,
                    'asset contribution to selection factor risk': asset_contrib_select_factor,
                    'asset contribution to selection specific risk': asset_contrib_select_specific,
                    'component_id': component_id
                }
                
                table_data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(table_data)
            
            # Ensure deterministic ordering by sorting by asset_name
            if not df.empty:
                df = df.sort_values('asset_name').reset_index(drop=True)
            
            logger.info(f"Created Brinson table with {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error creating Brinson table: {e}")
            # Return empty DataFrame with correct structure
            empty_df = pd.DataFrame(columns=[
                'asset_name', 'portfolio_weight', 'benchmark_weight', 'asset marginal',
                'asset contribution to risk', 'asset contribution to factor risk',
                'asset contribution to specific risk', 'asset contribution to allocation factor risk',
                'asset contribution to allocation specific risk', 'asset contribution to selection factor risk',
                'asset contribution to selection specific risk', 'component_id'
            ])
            return empty_df