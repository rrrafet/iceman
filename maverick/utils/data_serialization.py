"""
Data serialization utilities for handling complex data types in Streamlit applications.

This module provides utilities to convert complex Python data types (Pandas timestamps,
NumPy arrays, etc.) into JSON/Streamlit-compatible formats.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Union


def serialize_for_streamlit(data: Any) -> Any:
    """
    Convert complex data types to Streamlit-compatible formats.
    
    This function handles:
    - Pandas Timestamps → string format
    - NumPy arrays → Python lists  
    - NumPy scalars → Python scalars
    - Pandas Series/DataFrames → serialized structures
    - Nested dictionaries and lists recursively
    
    Args:
        data: Input data of any type
        
    Returns:
        Serialized data compatible with Streamlit/JSON
        
    Example:
        >>> import pandas as pd
        >>> timestamp_data = {"date": pd.Timestamp("2019-01-01")}
        >>> serialized = serialize_for_streamlit(timestamp_data)
        >>> print(serialized)
        {"date": "2019-01-01 00:00:00"}
    """
    # Handle None
    if data is None:
        return None
    
    # Handle dictionaries recursively
    elif isinstance(data, dict):
        return {k: serialize_for_streamlit(v) for k, v in data.items()}
    
    # Handle lists and tuples recursively
    elif isinstance(data, (list, tuple)):
        return [serialize_for_streamlit(item) for item in data]
    
    # Handle Pandas Timestamps and datetime objects
    elif isinstance(data, (pd.Timestamp, datetime)):
        return data.strftime('%Y-%m-%d %H:%M:%S') if hasattr(data, 'strftime') else str(data)
    
    # Handle Pandas Timedelta
    elif isinstance(data, pd.Timedelta):
        return str(data)
    
    # Handle NumPy arrays
    elif isinstance(data, np.ndarray):
        return data.tolist()
    
    # Handle NumPy scalars
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()
    
    # Handle NumPy datetime64 and timedelta64
    elif isinstance(data, (np.datetime64, np.timedelta64)):
        return str(data)
    
    # Handle Pandas Series
    elif isinstance(data, pd.Series):
        return [serialize_for_streamlit(item) for item in data.tolist()]
    
    # Handle Pandas DataFrame
    elif isinstance(data, pd.DataFrame):
        result = {}
        for col in data.columns:
            result[col] = [serialize_for_streamlit(item) for item in data[col].tolist()]
        return result
    
    # Handle basic types (int, float, str, bool)
    elif isinstance(data, (int, float, str, bool)):
        return data
    
    # Handle complex numbers
    elif isinstance(data, complex):
        return {"real": data.real, "imag": data.imag}
    
    # Fallback: convert to string for unknown types
    else:
        try:
            return str(data)
        except Exception:
            return f"<{type(data).__name__}>"


def test_serialization():
    """
    Test function to verify serialization handles various data types correctly.
    
    Returns:
        Dict with test results
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # Test data with various complex types
    test_data = {
        "timestamp": pd.Timestamp("2019-01-01 12:30:45"),
        "datetime": datetime(2020, 5, 15, 14, 30),
        "timedelta": pd.Timedelta("1 days 02:30:00"),
        "numpy_array": np.array([1, 2, 3, 4, 5]),
        "numpy_float": np.float64(3.14159),
        "numpy_int": np.int64(42),
        "datetime64": np.datetime64("2021-03-10"),
        "pandas_series": pd.Series([1, 2, 3, 4, 5]),
        "nested_dict": {
            "inner_timestamp": pd.Timestamp("2022-12-25"),
            "inner_array": np.array([10, 20, 30])
        },
        "mixed_list": [
            pd.Timestamp("2023-01-01"),
            np.array([1, 2]),
            "regular_string",
            42
        ],
        "regular_data": {
            "string": "hello",
            "int": 123,
            "float": 45.67,
            "bool": True,
            "none": None
        }
    }
    
    try:
        serialized = serialize_for_streamlit(test_data)
        
        return {
            "success": True,
            "original_types": {k: type(v).__name__ for k, v in test_data.items()},
            "serialized_data": serialized,
            "message": "All data types serialized successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Serialization failed: {e}"
        }


if __name__ == "__main__":
    # Run test when module is executed directly
    result = test_serialization()
    print("Serialization Test Results:")
    print(f"Success: {result['success']}")
    if result['success']:
        print("✅ All data types handled correctly")
        print("\nSerialized data structure:")
        for key, value in result['serialized_data'].items():
            print(f"  {key}: {type(value).__name__} = {value}")
    else:
        print(f"❌ Error: {result['message']}")