"""
Streamlit helper utilities for displaying data in Maverick application.

This module provides utilities specifically for displaying complex data types
in Streamlit without type errors.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from .data_serialization import serialize_for_streamlit


def safe_display_value(value: Any, key: str = None) -> None:
    """
    Safely display a value in Streamlit, converting complex types as needed.
    
    Args:
        value: Value to display
        key: Optional key for Streamlit widget
    """
    try:
        serialized_value = serialize_for_streamlit(value)
        st.write(serialized_value, key=key)
    except Exception as e:
        st.error(f"Error displaying value: {e}")
        st.write(f"Raw value type: {type(value).__name__}")


def safe_metric(label: str, value: Any, delta: Any = None, key: str = None) -> None:
    """
    Safely display a metric in Streamlit, handling complex data types.
    
    Args:
        label: Metric label
        value: Metric value (will be serialized if needed)
        delta: Optional delta value (will be serialized if needed)
        key: Optional key for Streamlit widget
    """
    try:
        safe_value = serialize_for_streamlit(value)
        safe_delta = serialize_for_streamlit(delta) if delta is not None else None
        st.metric(label, safe_value, safe_delta, key=key)
    except Exception as e:
        st.error(f"Error displaying metric '{label}': {e}")
        st.write(f"Value type: {type(value).__name__}, Value: {value}")


def safe_dataframe(df: Any, key: str = None, **kwargs) -> None:
    """
    Safely display a DataFrame in Streamlit, handling timestamp columns.
    
    Args:
        df: DataFrame or data to display
        key: Optional key for Streamlit widget
        **kwargs: Additional arguments for st.dataframe
    """
    try:
        if isinstance(df, pd.DataFrame):
            # Convert DataFrame columns that might have timestamps
            safe_df = df.copy()
            for col in safe_df.columns:
                if safe_df[col].dtype == 'datetime64[ns]':
                    safe_df[col] = safe_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(safe_df[col].iloc[0] if len(safe_df) > 0 else None, pd.Timestamp):
                    safe_df[col] = safe_df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(x) else '')
            st.dataframe(safe_df, key=key, **kwargs)
        else:
            # Try to convert to DataFrame if possible
            serialized_data = serialize_for_streamlit(df)
            if isinstance(serialized_data, dict):
                safe_df = pd.DataFrame(serialized_data)
                st.dataframe(safe_df, key=key, **kwargs)
            elif isinstance(serialized_data, list):
                st.dataframe(pd.DataFrame(serialized_data), key=key, **kwargs)
            else:
                st.write(serialized_data, key=key)
    except Exception as e:
        st.error(f"Error displaying dataframe: {e}")
        st.write(f"Data type: {type(df).__name__}")


def safe_json(data: Any, key: str = None) -> None:
    """
    Safely display data as JSON in Streamlit, serializing complex types.
    
    Args:
        data: Data to display as JSON
        key: Optional key for Streamlit widget
    """
    try:
        serialized_data = serialize_for_streamlit(data)
        st.json(serialized_data, key=key)
    except Exception as e:
        st.error(f"Error displaying JSON: {e}")
        st.write(f"Data type: {type(data).__name__}")


def safe_table(data: Any, key: str = None) -> None:
    """
    Safely display data as a table in Streamlit.
    
    Args:
        data: Data to display as table
        key: Optional key for Streamlit widget
    """
    try:
        if isinstance(data, dict):
            # Convert dict to DataFrame for table display
            serialized_data = serialize_for_streamlit(data)
            # Try to create a two-column table
            table_data = []
            for k, v in serialized_data.items():
                table_data.append({"Property": k, "Value": str(v)})
            st.table(pd.DataFrame(table_data), key=key)
        elif isinstance(data, (list, tuple)):
            serialized_data = serialize_for_streamlit(data)
            st.table(pd.DataFrame(serialized_data), key=key)
        else:
            serialized_data = serialize_for_streamlit(data)
            st.table(serialized_data, key=key)
    except Exception as e:
        st.error(f"Error displaying table: {e}")
        st.write(f"Data type: {type(data).__name__}")


def create_safe_plotly_data(data: Any) -> Any:
    """
    Create Plotly-safe data by serializing complex types.
    
    Args:
        data: Data to make Plotly-compatible
        
    Returns:
        Plotly-compatible data
    """
    return serialize_for_streamlit(data)


def display_type_safe_info(data: Any, title: str = "Data Information") -> None:
    """
    Display information about data in a type-safe manner.
    
    Args:
        data: Data to analyze and display
        title: Title for the information section
    """
    with st.expander(title):
        st.write(f"**Type:** {type(data).__name__}")
        
        if isinstance(data, dict):
            st.write(f"**Keys:** {len(data)}")
            for key, value in list(data.items())[:5]:  # Show first 5 keys
                st.write(f"- {key}: {type(value).__name__}")
            if len(data) > 5:
                st.write(f"... and {len(data) - 5} more keys")
        
        elif isinstance(data, (list, tuple)):
            st.write(f"**Length:** {len(data)}")
            if len(data) > 0:
                st.write(f"**First item type:** {type(data[0]).__name__}")
        
        elif isinstance(data, pd.DataFrame):
            st.write(f"**Shape:** {data.shape}")
            st.write(f"**Columns:** {list(data.columns)}")
        
        elif isinstance(data, pd.Series):
            st.write(f"**Length:** {len(data)}")
            st.write(f"**Dtype:** {data.dtype}")
        
        # Try to safely display a sample
        try:
            sample_data = serialize_for_streamlit(data)
            if isinstance(sample_data, dict) and len(sample_data) > 3:
                # Show only first few items of large dicts
                sample_keys = list(sample_data.keys())[:3]
                sample_dict = {k: sample_data[k] for k in sample_keys}
                st.write("**Sample:**", sample_dict)
                st.write(f"... ({len(sample_data) - 3} more items)")
            else:
                st.write("**Content:**", sample_data)
        except Exception:
            st.write("**Content:** <Unable to display safely>")