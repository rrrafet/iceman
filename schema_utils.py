"""
Schema Utilities - Helper functions for risk data processing and validation.

This module provides utility functions for converting, processing, and validating
risk data extracted from portfolio visitors and decomposers.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def array_to_named_dict(
    array: np.ndarray,
    names: List[str],
    fallback_prefix: str = "Item"
) -> Dict[str, float]:
    """
    Convert numpy array to named dictionary using provided names.
    
    Parameters
    ----------
    array : np.ndarray
        Array to convert
    names : List[str]
        Names to use for dictionary keys
    fallback_prefix : str, default "Item"
        Prefix for fallback names if names list is insufficient
        
    Returns
    -------
    Dict[str, float]
        Named dictionary with array values
    """
    if array is None:
        return {}
    
    array = np.asarray(array).flatten()
    
    if len(array) == 0:
        return {}
    
    # Ensure we have enough names
    if len(names) < len(array):
        # Extend names list with fallback names
        additional_names = [f"{fallback_prefix}_{i+1}" for i in range(len(names), len(array))]
        names = list(names) + additional_names
        logger.warning(f"Insufficient names provided ({len(names)} names for {len(array)} array elements). "
                      f"Using fallback names with prefix '{fallback_prefix}'")
    
    # Create dictionary
    return {names[i]: float(array[i]) for i in range(len(array))}


def matrix_to_nested_dict(
    matrix: np.ndarray,
    row_names: List[str],
    col_names: List[str],
    fallback_row_prefix: str = "Row",
    fallback_col_prefix: str = "Col"
) -> Dict[str, Dict[str, float]]:
    """
    Convert 2D numpy array to nested dictionary with named rows and columns.
    
    Parameters
    ----------
    matrix : np.ndarray
        2D array to convert
    row_names : List[str]
        Names for matrix rows
    col_names : List[str]
        Names for matrix columns
    fallback_row_prefix : str, default "Row"
        Prefix for fallback row names
    fallback_col_prefix : str, default "Col"
        Prefix for fallback column names
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary with named rows and columns
    """
    if matrix is None:
        return {}
    
    matrix = np.asarray(matrix)
    
    if matrix.size == 0:
        return {}
    
    # Ensure matrix is 2D
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim > 2:
        logger.warning(f"Matrix has {matrix.ndim} dimensions, flattening to 2D")
        matrix = matrix.reshape(matrix.shape[0], -1)
    
    rows, cols = matrix.shape
    
    # Ensure we have enough names
    if len(row_names) < rows:
        additional_row_names = [f"{fallback_row_prefix}_{i+1}" for i in range(len(row_names), rows)]
        row_names = list(row_names) + additional_row_names
        logger.warning(f"Insufficient row names provided. Using fallback names with prefix '{fallback_row_prefix}'")
    
    if len(col_names) < cols:
        additional_col_names = [f"{fallback_col_prefix}_{i+1}" for i in range(len(col_names), cols)]
        col_names = list(col_names) + additional_col_names
        logger.warning(f"Insufficient column names provided. Using fallback names with prefix '{fallback_col_prefix}'")
    
    # Create nested dictionary
    return {
        row_names[i]: {
            col_names[j]: float(matrix[i, j]) for j in range(cols)
        } for i in range(rows)
    }


def extract_property_safely(
    obj: Any,
    property_name: str,
    default_value: Any = None,
    log_errors: bool = True
) -> Any:
    """
    Safely extract a property from an object with error handling.
    
    Parameters
    ----------
    obj : Any
        Object to extract property from
    property_name : str
        Name of property to extract
    default_value : Any, optional
        Default value if extraction fails
    log_errors : bool, default True
        Whether to log extraction errors
        
    Returns
    -------
    Any
        Property value or default value
    """
    if obj is None:
        return default_value
    
    try:
        if hasattr(obj, property_name):
            value = getattr(obj, property_name)
            # Handle callable properties
            if callable(value):
                return value()
            return value
        else:
            if log_errors:
                logger.debug(f"Property '{property_name}' not found in object of type {type(obj).__name__}")
            return default_value
    except Exception as e:
        if log_errors:
            logger.warning(f"Error extracting property '{property_name}': {e}")
        return default_value


def validate_array_dimensions(
    array: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    name: str = "array"
) -> Dict[str, Any]:
    """
    Validate array dimensions and properties.
    
    Parameters
    ----------
    array : np.ndarray
        Array to validate
    expected_shape : Tuple[int, ...], optional
        Expected exact shape
    min_dims : int, optional
        Minimum number of dimensions
    max_dims : int, optional
        Maximum number of dimensions
    name : str, default "array"
        Name for logging purposes
        
    Returns
    -------
    Dict[str, Any]
        Validation results
    """
    if array is None:
        return {
            'valid': False,
            'issues': [f"{name} is None"],
            'array_info': None
        }
    
    issues = []
    
    array = np.asarray(array)
    
    # Check for finite values
    if not np.all(np.isfinite(array)):
        nan_count = np.sum(np.isnan(array))
        inf_count = np.sum(np.isinf(array))
        issues.append(f"{name} contains {nan_count} NaN and {inf_count} infinite values")
    
    # Check dimensions
    if min_dims is not None and array.ndim < min_dims:
        issues.append(f"{name} has {array.ndim} dimensions, minimum required: {min_dims}")
    
    if max_dims is not None and array.ndim > max_dims:
        issues.append(f"{name} has {array.ndim} dimensions, maximum allowed: {max_dims}")
    
    # Check exact shape if specified
    if expected_shape is not None and array.shape != expected_shape:
        issues.append(f"{name} shape {array.shape} does not match expected {expected_shape}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'array_info': {
            'shape': array.shape,
            'dtype': str(array.dtype),
            'size': array.size,
            'has_nan': np.any(np.isnan(array)),
            'has_inf': np.any(np.isinf(array)),
            'min': float(np.min(array)) if array.size > 0 and np.all(np.isfinite(array)) else None,
            'max': float(np.max(array)) if array.size > 0 and np.all(np.isfinite(array)) else None
        }
    }


def process_time_series_data(
    time_series: Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]],
    component_id: str
) -> Dict[str, pd.DataFrame]:
    """
    Process time series data into standardized format.
    
    Parameters
    ----------
    time_series : Union[pd.Series, pd.DataFrame, Dict[str, pd.Series]]
        Time series data in various formats
    component_id : str
        Component identifier for naming
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Standardized time series data
    """
    result = {}
    
    try:
        if isinstance(time_series, pd.Series):
            # Single series - convert to DataFrame
            result[f"{component_id}_returns"] = time_series.to_frame(component_id)
        
        elif isinstance(time_series, pd.DataFrame):
            # DataFrame - use as-is
            result[f"{component_id}_returns"] = time_series
        
        elif isinstance(time_series, dict):
            # Dictionary of series
            for name, series in time_series.items():
                if isinstance(series, pd.Series):
                    result[f"{component_id}_{name}"] = series.to_frame(f"{component_id}_{name}")
                elif isinstance(series, pd.DataFrame):
                    result[f"{component_id}_{name}"] = series
                else:
                    logger.warning(f"Unsupported time series type for {name}: {type(series)}")
        
        else:
            logger.warning(f"Unsupported time series format: {type(time_series)}")
    
    except Exception as e:
        logger.error(f"Error processing time series data for {component_id}: {e}")
    
    return result


def calculate_risk_percentages(
    total_risk: float,
    factor_risk: float,
    specific_risk: float,
    tolerance: float = 1e-6
) -> Dict[str, float]:
    """
    Calculate risk percentage breakdown with validation.
    
    Parameters
    ----------
    total_risk : float
        Total portfolio risk
    factor_risk : float
        Factor risk contribution
    specific_risk : float
        Specific risk contribution
    tolerance : float, default 1e-6
        Tolerance for validation checks
        
    Returns
    -------
    Dict[str, float]
        Risk percentages and validation metrics
    """
    result = {
        'factor_risk_percentage': 0.0,
        'specific_risk_percentage': 0.0,
        'total_percentage': 0.0,
        'euler_validation': True,
        'euler_error': 0.0
    }
    
    try:
        # Calculate percentages
        if total_risk > 0:
            result['factor_risk_percentage'] = (factor_risk / total_risk) * 100
            result['specific_risk_percentage'] = (specific_risk / total_risk) * 100
            result['total_percentage'] = result['factor_risk_percentage'] + result['specific_risk_percentage']
        
        # Validate Euler identity
        computed_total = factor_risk + specific_risk
        euler_error = abs(computed_total - total_risk)
        result['euler_error'] = euler_error
        result['euler_validation'] = euler_error < tolerance
        
        if not result['euler_validation']:
            logger.warning(f"Euler identity validation failed: error = {euler_error:.6f}")
    
    except Exception as e:
        logger.error(f"Error calculating risk percentages: {e}")
    
    return result


def validate_weight_vector(
    weights: np.ndarray,
    weight_type: str = "portfolio",
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Validate weight vector properties.
    
    Parameters
    ----------
    weights : np.ndarray
        Weight vector to validate
    weight_type : str, default "portfolio"
        Type of weights ('portfolio', 'benchmark', 'active')
    tolerance : float, default 0.01
        Tolerance for validation checks
        
    Returns
    -------
    Dict[str, Any]
        Validation results
    """
    if weights is None:
        return {
            'valid': False,
            'issues': [f"{weight_type} weights are None"],
            'weight_sum': None
        }
    
    weights = np.asarray(weights).flatten()
    issues = []
    
    # Check for finite values
    if not np.all(np.isfinite(weights)):
        issues.append(f"{weight_type} weights contain NaN or infinite values")
    
    # Check weight sum based on type
    weight_sum = np.sum(weights)
    if weight_type in ['portfolio', 'benchmark']:
        if abs(weight_sum - 1.0) > tolerance:
            issues.append(f"{weight_type} weights sum to {weight_sum:.6f}, expected H 1.0")
    elif weight_type == 'active':
        if abs(weight_sum) > tolerance:
            issues.append(f"Active weights sum to {weight_sum:.6f}, expected H 0.0")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'weight_sum': float(weight_sum),
        'weight_stats': {
            'min': float(np.min(weights)),
            'max': float(np.max(weights)),
            'mean': float(np.mean(weights)),
            'std': float(np.std(weights)),
            'count': len(weights)
        }
    }


def merge_nested_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two nested dictionaries.
    
    Parameters
    ----------
    dict1 : Dict[str, Any]
        First dictionary
    dict2 : Dict[str, Any]
        Second dictionary (takes precedence)
        
    Returns
    -------
    Dict[str, Any]
        Merged dictionary
    """
    if dict1 is None:
        return dict2.copy() if dict2 else {}
    if dict2 is None:
        return dict1.copy()
    
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_nested_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def clean_nan_values(data: Dict[str, Any], replacement: Any = 0.0) -> Dict[str, Any]:
    """
    Clean NaN values from nested dictionary data.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Data dictionary to clean
    replacement : Any, default 0.0
        Value to replace NaN with
        
    Returns
    -------
    Dict[str, Any]
        Cleaned data dictionary
    """
    if not isinstance(data, dict):
        return data
    
    result = {}
    
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = clean_nan_values(value, replacement)
        elif isinstance(value, (np.ndarray, list)):
            value_array = np.asarray(value)
            if np.any(np.isnan(value_array)):
                value_array = np.where(np.isnan(value_array), replacement, value_array)
                result[key] = value_array.tolist() if isinstance(value, list) else value_array
            else:
                result[key] = value
        elif isinstance(value, float) and np.isnan(value):
            result[key] = replacement
        else:
            result[key] = value
    
    return result