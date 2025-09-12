"""
Central Resampling Service for Spark Platform

This module provides unified resampling and frequency management for all return data
processing, consolidating logic that was previously scattered across multiple data providers.

Key Features:
- Single source of truth for all resampling operations
- Consistent compound return calculations
- Integrated frequency management and validation
- Support for Series, DataFrame, and nested dictionary data structures
- Comprehensive error handling and logging
"""

import pandas as pd
import numpy as np
import logging
from typing import Union, Optional, Dict, Any, List
from datetime import datetime
from spark.core.mappers import frequency_to_multiplier
from spark.risk.annualizer import RiskAnnualizer

logger = logging.getLogger(__name__)


class FrequencyValidator:
    """Validates and normalizes frequency specifications."""
    
    SUPPORTED_FREQUENCIES = ["D", "B", "W-FRI", "ME", "Q", "A"]
    NATIVE_FREQUENCIES = ["D", "B"]
    
    @classmethod
    def validate_frequency(cls, frequency: str) -> str:
        """
        Validate and normalize frequency string.
        
        Args:
            frequency: Frequency string to validate
            
        Returns:
            Normalized frequency string
            
        Raises:
            ValueError: If frequency is not supported
        """
        if frequency not in cls.SUPPORTED_FREQUENCIES:
            raise ValueError(
                f"Unsupported frequency: {frequency}. "
                f"Supported frequencies: {cls.SUPPORTED_FREQUENCIES}"
            )
        return frequency
    
    @classmethod
    def is_native_frequency(cls, frequency: str) -> bool:
        """Check if frequency is native (no resampling needed)."""
        return frequency in cls.NATIVE_FREQUENCIES
    
    @classmethod
    def requires_resampling(cls, current_freq: str, target_freq: str) -> bool:
        """Check if resampling is required between frequencies."""
        return current_freq != target_freq and not cls.is_native_frequency(target_freq)


class ResamplingService:
    """
    Unified resampling service that consolidates all frequency conversion logic.
    
    This service replaces the scattered resampling methods across data providers
    with a single, well-tested implementation that handles all data types consistently.
    """
    
    def __init__(self, native_frequency: str = "B"):
        """
        Initialize resampling service.
        
        Args:
            native_frequency: The native frequency of source data
        """
        self.native_frequency = FrequencyValidator.validate_frequency(native_frequency)
        self.current_frequency = native_frequency
        logger.info(f"ResamplingService initialized with native frequency: {native_frequency}")
    
    def set_target_frequency(self, target_frequency: str) -> bool:
        """
        Set target frequency for resampling operations.
        
        Args:
            target_frequency: Target frequency for resampling
            
        Returns:
            True if frequency changed, False otherwise
        """
        validated_freq = FrequencyValidator.validate_frequency(target_frequency)
        
        if validated_freq != self.current_frequency:
            old_freq = self.current_frequency
            self.current_frequency = validated_freq
            logger.info(f"Target frequency changed: {old_freq} -> {validated_freq}")
            return True
        return False
    
    def requires_resampling(self) -> bool:
        """Check if current configuration requires resampling."""
        return FrequencyValidator.requires_resampling(
            self.native_frequency, self.current_frequency
        )
    
    def resample_series(self, series: pd.Series, validate_data: bool = True) -> pd.Series:
        """
        Resample returns time series using compound return calculation.
        
        Args:
            series: Time series to resample
            validate_data: Whether to validate input data
            
        Returns:
            Resampled series or original if no resampling needed
        """
        if not self.requires_resampling() or series.empty:
            return series
        
        if validate_data:
            self._validate_series_input(series)
        
        try:
            # Use compound return calculation: (1+r).prod() - 1
            resampled = series.resample(self.current_frequency).apply(
                lambda x: (1 + x).prod() - 1 if not x.empty else np.nan
            )
            resampled.name = series.name
            result = resampled.dropna()
            
            logger.debug(
                f"Resampled series '{series.name}' from {len(series)} to {len(result)} "
                f"observations at {self.current_frequency} frequency"
            )
            return result
            
        except Exception as e:
            logger.error(f"Error resampling series '{series.name}': {e}")
            raise ResamplingError(f"Failed to resample series: {e}") from e
    
    def resample_dataframe(self, df: pd.DataFrame, validate_data: bool = True) -> pd.DataFrame:
        """
        Resample returns DataFrame using compound return calculation.
        
        Args:
            df: DataFrame to resample
            validate_data: Whether to validate input data
            
        Returns:
            Resampled DataFrame or original if no resampling needed
        """
        if not self.requires_resampling() or df.empty:
            return df
        
        if validate_data:
            self._validate_dataframe_input(df)
        
        try:
            # Apply compound return calculation to each column
            resampled = df.resample(self.current_frequency).apply(
                lambda x: (1 + x).prod() - 1 if not x.empty else np.nan
            )
            result = resampled.dropna(how='all')
            
            logger.debug(
                f"Resampled DataFrame from {len(df)} to {len(result)} "
                f"observations at {self.current_frequency} frequency"
            )
            return result
            
        except Exception as e:
            logger.error(f"Error resampling DataFrame: {e}")
            raise ResamplingError(f"Failed to resample DataFrame: {e}") from e
    
    def resample_returns_dict(self, returns_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Resample dictionary of return series.
        
        Args:
            returns_dict: Dictionary mapping component IDs to return series
            
        Returns:
            Dictionary with resampled series
        """
        if not returns_dict or not self.requires_resampling():
            return returns_dict
        
        resampled_dict = {}
        for component_id, series in returns_dict.items():
            try:
                resampled_dict[component_id] = self.resample_series(series, validate_data=False)
            except Exception as e:
                logger.warning(f"Failed to resample series for {component_id}: {e}")
                resampled_dict[component_id] = series  # Keep original on error
        
        return resampled_dict
    
    def align_and_resample_data(
        self, 
        data_sources: Dict[str, Union[pd.Series, pd.DataFrame]],
        alignment_method: str = "intersection"
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Align multiple data sources to common date index and resample consistently.
        
        Args:
            data_sources: Dictionary of data sources to align and resample
            alignment_method: Method for date alignment ('intersection' or 'union')
            
        Returns:
            Dictionary with aligned and resampled data
        """
        if not data_sources:
            return {}
        
        logger.info(f"Aligning {len(data_sources)} data sources using {alignment_method} method")
        
        # First, resample all data sources
        resampled_sources = {}
        for name, data in data_sources.items():
            if isinstance(data, pd.Series):
                resampled_sources[name] = self.resample_series(data)
            elif isinstance(data, pd.DataFrame):
                resampled_sources[name] = self.resample_dataframe(data)
            else:
                logger.warning(f"Unsupported data type for {name}: {type(data)}")
                resampled_sources[name] = data
        
        # Then align to common date index
        aligned_sources = self._align_data_sources(resampled_sources, alignment_method)
        
        return aligned_sources
    
    def calculate_empirical_volatility(
        self, 
        returns: pd.Series, 
        annualize: bool = True,
        window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate empirical volatility from return series.
        
        Args:
            returns: Return time series
            annualize: Whether to annualize the volatility
            window: Rolling window size (None for full period)
            
        Returns:
            Volatility (scalar or series if window specified)
        """
        if returns.empty:
            return np.nan if window is None else pd.Series(dtype=float)
        
        # Resample if needed
        resampled_returns = self.resample_series(returns)
        
        if window is None:
            # Calculate full-period volatility
            volatility = resampled_returns.std()
            if annualize and not np.isnan(volatility):
                volatility = RiskAnnualizer.annualize_volatility(
                    volatility, self.current_frequency
                )
            return volatility
        else:
            # Calculate rolling volatility
            rolling_vol = resampled_returns.rolling(window=window).std()
            if annualize:
                rolling_vol = rolling_vol.apply(
                    lambda x: RiskAnnualizer.annualize_volatility(x, self.current_frequency)
                    if not np.isnan(x) else np.nan
                )
            return rolling_vol
    
    def validate_data_alignment(
        self, 
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: Optional[pd.DataFrame] = None,
        tolerance_days: int = 5
    ) -> Dict[str, Any]:
        """
        Validate that return data sources are properly aligned.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series  
            factor_returns: Factor returns DataFrame (optional)
            tolerance_days: Maximum allowed misalignment in days
            
        Returns:
            Validation results dictionary
        """
        validation_result = {
            "aligned": True,
            "issues": [],
            "date_ranges": {},
            "overlap_periods": {},
            "recommendations": []
        }
        
        # Check date ranges
        data_sources = {
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns
        }
        if factor_returns is not None and not factor_returns.empty:
            data_sources["factors"] = factor_returns
        
        for name, data in data_sources.items():
            if not data.empty:
                validation_result["date_ranges"][name] = {
                    "start": data.index.min(),
                    "end": data.index.max(),
                    "count": len(data)
                }
        
        # Check alignment between portfolio and benchmark
        if not portfolio_returns.empty and not benchmark_returns.empty:
            portfolio_aligned, benchmark_aligned = portfolio_returns.align(benchmark_returns)
            common_dates = len(portfolio_aligned.dropna().index.intersection(
                benchmark_aligned.dropna().index
            ))
            
            validation_result["overlap_periods"]["portfolio_benchmark"] = common_dates
            
            if common_dates == 0:
                validation_result["aligned"] = False
                validation_result["issues"].append("No overlapping dates between portfolio and benchmark")
        
        # Add recommendations based on issues
        if not validation_result["aligned"]:
            validation_result["recommendations"].append(
                "Consider adjusting date ranges to ensure proper data alignment"
            )
        
        return validation_result
    
    def get_frequency_info(self) -> Dict[str, Any]:
        """Get detailed frequency configuration information."""
        return {
            "native_frequency": self.native_frequency,
            "current_frequency": self.current_frequency,
            "requires_resampling": self.requires_resampling(),
            "annualization_multiplier": frequency_to_multiplier.get(
                self.current_frequency.upper(), 1.0
            ),
            "periods_per_year": RiskAnnualizer.get_periods_per_year(self.current_frequency)
        }
    
    def _validate_series_input(self, series: pd.Series) -> None:
        """Validate Series input for resampling."""
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ResamplingError("Series must have DatetimeIndex for resampling")
        
        if series.index.duplicated().any():
            raise ResamplingError("Series index contains duplicate dates")
    
    def _validate_dataframe_input(self, df: pd.DataFrame) -> None:
        """Validate DataFrame input for resampling."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ResamplingError("DataFrame must have DatetimeIndex for resampling")
        
        if df.index.duplicated().any():
            raise ResamplingError("DataFrame index contains duplicate dates")
    
    def _align_data_sources(
        self,
        data_sources: Dict[str, Union[pd.Series, pd.DataFrame]],
        method: str
    ) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Align multiple data sources to common date index."""
        if not data_sources or len(data_sources) <= 1:
            return data_sources
        
        # Get all date indices
        all_indices = []
        for name, data in data_sources.items():
            if hasattr(data, 'index') and len(data) > 0:
                all_indices.append(data.index)
        
        if not all_indices:
            return data_sources
        
        # Determine common index based on method
        if method == "intersection":
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)
        elif method == "union":
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.union(idx)
        else:
            raise ValueError(f"Unknown alignment method: {method}")
        
        # Reindex all data sources
        aligned_sources = {}
        for name, data in data_sources.items():
            if isinstance(data, (pd.Series, pd.DataFrame)):
                aligned_sources[name] = data.reindex(common_index)
            else:
                aligned_sources[name] = data
        
        logger.debug(f"Aligned {len(data_sources)} sources to {len(common_index)} common dates")
        return aligned_sources


class ResamplingError(Exception):
    """Exception raised for resampling-related errors."""
    pass


def create_resampling_service(native_frequency: str = "B") -> ResamplingService:
    """
    Factory function to create a ResamplingService instance.
    
    Args:
        native_frequency: Native data frequency
        
    Returns:
        Configured ResamplingService instance
    """
    return ResamplingService(native_frequency=native_frequency)