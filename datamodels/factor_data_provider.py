"""
Factor Data Provider for portfolio risk analysis system.
Handles loading and processing factor returns from parquet files.
"""

from typing import List, Tuple, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FactorDataProvider:
    """
    Provides comprehensive access to factor returns data from parquet files.
    
    Handles factor data in long format with columns: date, factor_name, return_value, riskmodel_code
    """
    
    def __init__(self, factor_returns_path: str):
        """
        Initialize factor data provider.
        
        Args:
            factor_returns_path: Path to factor returns parquet file
        """
        self.factor_returns_path = Path(factor_returns_path)
        self._data: Optional[pd.DataFrame] = None
        
        # Frequency management for data providers
        self.frequency_manager = None
        self.current_frequency = "B"  # Default to business daily
        
        self._load_data()
    
    def set_frequency_manager(self, frequency_manager):
        """
        Set frequency manager to coordinate frequency with DataAccessService.
        
        Args:
            frequency_manager: FrequencyManager instance from DataAccessService
        """
        self.frequency_manager = frequency_manager
        if frequency_manager:
            self.current_frequency = frequency_manager.current_frequency
            logger.info(f"FactorDataProvider frequency set to: {self.current_frequency}")
    
    def _resample_returns_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample returns DataFrame using compound return calculation.
        
        Args:
            df: Returns DataFrame to resample
            
        Returns:
            Resampled DataFrame or original DataFrame if no resampling needed
        """
        if not self.frequency_manager or not self.frequency_manager.is_resampled or df.empty:
            return df
        
        freq = self.frequency_manager.current_frequency
        try:
            # Apply compound return calculation to each column
            resampled = df.resample(freq).apply(lambda x: (1 + x).prod() - 1)
            logger.debug(f"FactorDataProvider resampled DataFrame from {len(df)} to {len(resampled)} observations at {freq}")
            return resampled.dropna()
            
        except Exception as e:
            logger.error(f"Error resampling DataFrame in FactorDataProvider: {e}")
            return df
    
    def _load_data(self) -> None:
        """Load factor returns data from parquet file."""
        try:
            if not self.factor_returns_path.exists():
                raise FileNotFoundError(f"Factor returns file not found: {self.factor_returns_path}")
            
            self._data = pd.read_parquet(self.factor_returns_path)
            
            # Validate required columns
            required_columns = {'date', 'factor_name', 'return_value', 'riskmodel_code'}
            if not required_columns.issubset(self._data.columns):
                missing = required_columns - set(self._data.columns)
                raise ValueError(f"Missing required columns in factor data: {missing}")
            
            # Convert date column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(self._data['date']):
                self._data['date'] = pd.to_datetime(self._data['date'])
            
            # Sort by date and factor for consistency
            self._data = self._data.sort_values(['date', 'factor_name', 'riskmodel_code'])
            
            logger.info(f"Loaded factor data: {len(self._data)} records, "
                       f"{self._data['factor_name'].nunique()} factors, "
                       f"{self._data['riskmodel_code'].nunique()} risk models")
                       
        except Exception as e:
            logger.error(f"Failed to load factor data from {self.factor_returns_path}: {e}")
            raise
    
    def load_factor_returns(self, risk_model_code: str) -> pd.DataFrame:
        """
        Load and filter factor returns by risk model code.
        
        Args:
            risk_model_code: Risk model code to filter by
            
        Returns:
            DataFrame with factor returns for specified model
        """
        if self._data is None:
            raise RuntimeError("Factor data not loaded")
        
        filtered_data = self._data[self._data['riskmodel_code'] == risk_model_code].copy()
        
        if filtered_data.empty:
            logger.warning(f"No factor data found for risk model: {risk_model_code}")
            return pd.DataFrame()
        
        logger.debug(f"Loaded {len(filtered_data)} factor return records for model {risk_model_code}")
        return filtered_data
    
    def get_factor_returns_long(self) -> pd.DataFrame:
        """
        Get raw factor returns data in long format.
        
        Returns:
            DataFrame with all factor returns in long format
        """
        if self._data is None:
            raise RuntimeError("Factor data not loaded")
        
        return self._data.copy()
    
    def get_factor_returns_wide(self, risk_model_code: str) -> pd.DataFrame:
        """
        Get factor returns pivoted to wide format for analysis.
        
        Args:
            risk_model_code: Risk model code to filter by
            
        Returns:
            DataFrame with dates as index and factors as columns
        """
        factor_data = self.load_factor_returns(risk_model_code)
        
        if factor_data.empty:
            return pd.DataFrame()
        
        # Pivot to wide format
        wide_data = factor_data.pivot_table(
            index='date',
            columns='factor_name',
            values='return_value',
            aggfunc='first'  # Handle duplicates by taking first value
        )
        
        # Fill any NaN values with 0 (common for factor models)
        wide_data = wide_data.fillna(0.0)
        
        logger.debug(f"Created wide factor returns: {wide_data.shape[0]} dates, {wide_data.shape[1]} factors")
        # Apply resampling if frequency manager is set
        return self._resample_returns_dataframe(wide_data)
    
    def get_available_risk_models(self) -> List[str]:
        """
        Get list of available risk model codes.
        
        Returns:
            List of unique risk model codes
        """
        if self._data is None:
            raise RuntimeError("Factor data not loaded")
        
        models = sorted(self._data['riskmodel_code'].unique())
        logger.debug(f"Available risk models: {models}")
        return models
    
    def get_factor_names(self, risk_model_code: str) -> List[str]:
        """
        Get factor names for a specific risk model.
        
        Args:
            risk_model_code: Risk model code
            
        Returns:
            List of factor names for the specified model
        """
        if self._data is None:
            raise RuntimeError("Factor data not loaded")
        
        model_data = self._data[self._data['riskmodel_code'] == risk_model_code]
        factors = sorted(model_data['factor_name'].unique())
        
        logger.debug(f"Factors for model {risk_model_code}: {len(factors)} factors")
        return factors
    
    def get_factor_time_series(self, factor_name: str, risk_model_code: str) -> pd.Series:
        """
        Get time series for a single factor.
        
        Args:
            factor_name: Name of the factor
            risk_model_code: Risk model code
            
        Returns:
            Series with dates as index and factor returns as values
        """
        if self._data is None:
            raise RuntimeError("Factor data not loaded")
        
        factor_data = self._data[
            (self._data['factor_name'] == factor_name) & 
            (self._data['riskmodel_code'] == risk_model_code)
        ].copy()
        
        if factor_data.empty:
            logger.warning(f"No data found for factor {factor_name} in model {risk_model_code}")
            return pd.Series(dtype=float, name=factor_name)
        
        # Create time series
        series = factor_data.set_index('date')['return_value'].sort_index()
        series.name = factor_name
        
        logger.debug(f"Created time series for {factor_name}: {len(series)} observations")
        return series
    
    def get_factor_correlation_matrix(self, risk_model_code: str) -> pd.DataFrame:
        """
        Calculate correlation matrix for factors in a risk model.
        
        Args:
            risk_model_code: Risk model code
            
        Returns:
            DataFrame with factor correlation matrix
        """
        wide_data = self.get_factor_returns_wide(risk_model_code)
        
        if wide_data.empty:
            logger.warning(f"Cannot compute correlation matrix: no data for model {risk_model_code}")
            return pd.DataFrame()
        
        correlation_matrix = wide_data.corr()
        
        logger.debug(f"Computed correlation matrix for model {risk_model_code}: "
                    f"{correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}")
        return correlation_matrix
    
    def get_date_range(self, risk_model_code: str) -> Tuple[datetime, datetime]:
        """
        Get date range for a risk model.
        
        Args:
            risk_model_code: Risk model code
            
        Returns:
            Tuple of (start_date, end_date)
        """
        if self._data is None:
            raise RuntimeError("Factor data not loaded")
        
        model_data = self._data[self._data['riskmodel_code'] == risk_model_code]
        
        if model_data.empty:
            raise ValueError(f"No data found for risk model: {risk_model_code}")
        
        start_date = model_data['date'].min().to_pydatetime()
        end_date = model_data['date'].max().to_pydatetime()
        
        logger.debug(f"Date range for model {risk_model_code}: {start_date} to {end_date}")
        return start_date, end_date
    
    def validate_data_completeness(self, risk_model_code: str) -> Dict[str, any]:
        """
        Validate data completeness for a risk model.
        
        Args:
            risk_model_code: Risk model code to validate
            
        Returns:
            Dictionary with validation results
        """
        if self._data is None:
            return {"valid": False, "error": "No data loaded"}
        
        model_data = self._data[self._data['riskmodel_code'] == risk_model_code]
        
        if model_data.empty:
            return {"valid": False, "error": f"No data for risk model: {risk_model_code}"}
        
        # Basic validation metrics
        total_records = len(model_data)
        unique_dates = model_data['date'].nunique()
        unique_factors = model_data['factor_name'].nunique()
        missing_returns = model_data['return_value'].isna().sum()
        date_range = self.get_date_range(risk_model_code)
        
        # Expected records (dates × factors)
        expected_records = unique_dates * unique_factors
        completeness_ratio = total_records / expected_records if expected_records > 0 else 0
        
        validation_result = {
            "valid": missing_returns == 0 and completeness_ratio >= 0.95,
            "risk_model_code": risk_model_code,
            "total_records": total_records,
            "unique_dates": unique_dates,
            "unique_factors": unique_factors,
            "missing_returns": missing_returns,
            "date_range": date_range,
            "completeness_ratio": completeness_ratio,
            "expected_records": expected_records
        }
        
        logger.info(f"Validation for model {risk_model_code}: "
                   f"{'PASS' if validation_result['valid'] else 'FAIL'} "
                   f"({completeness_ratio:.1%} complete)")
        
        return validation_result