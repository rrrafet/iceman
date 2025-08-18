"""
Transformable Abstract Base Class for DataFrame Operations

Provides a fluent interface for filtering pandas DataFrames with method chaining.
All filtering methods return self to enable chaining operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Union, List, Optional
import pandas as pd


class Transformable(ABC):
    """
    Abstract base class for transforming pandas DataFrames with fluent interface.
    
    All filtering methods accept keyword arguments where the key represents
    a column name and the value is the filtering criteria.
    
    Example:
        transformer = ConcreteTransformable(df)
        result = transformer.eq(date='2025-01-01').gt(price=100).data
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a pandas DataFrame.
        
        Args:
            data: Input pandas DataFrame to transform
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        self.data = data.copy()
    
    def eq(self, **kwargs) -> 'Transformable':
        """
        Filter rows where columns equal specified values.
        
        Args:
            **kwargs: Column-value pairs for equality filtering
            
        Returns:
            Self for method chaining
            
        Example:
            .eq(status='active', category='A')
        """
        for column, value in kwargs.items():
            if column not in self.data.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame")
            self.data = self.data[self.data[column] == value]
        return self
    
    def ne(self, **kwargs) -> 'Transformable':
        """
        Filter rows where columns do not equal specified values.
        
        Args:
            **kwargs: Column-value pairs for inequality filtering
            
        Returns:
            Self for method chaining
        """
        for column, value in kwargs.items():
            if column not in self.data.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame")
            self.data = self.data[self.data[column] != value]
        return self
    
    def gt(self, **kwargs) -> 'Transformable':
        """
        Filter rows where columns are greater than specified values.
        
        Args:
            **kwargs: Column-value pairs for greater than filtering
            
        Returns:
            Self for method chaining
        """
        for column, value in kwargs.items():
            if column not in self.data.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame")
            self.data = self.data[self.data[column] > value]
        return self
    
    def lt(self, **kwargs) -> 'Transformable':
        """
        Filter rows where columns are less than specified values.
        
        Args:
            **kwargs: Column-value pairs for less than filtering
            
        Returns:
            Self for method chaining
        """
        for column, value in kwargs.items():
            if column not in self.data.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame")
            self.data = self.data[self.data[column] < value]
        return self
    
    def gte(self, **kwargs) -> 'Transformable':
        """
        Filter rows where columns are greater than or equal to specified values.
        
        Args:
            **kwargs: Column-value pairs for greater than or equal filtering
            
        Returns:
            Self for method chaining
        """
        for column, value in kwargs.items():
            if column not in self.data.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame")
            self.data = self.data[self.data[column] >= value]
        return self
    
    def lte(self, **kwargs) -> 'Transformable':
        """
        Filter rows where columns are less than or equal to specified values.
        
        Args:
            **kwargs: Column-value pairs for less than or equal filtering
            
        Returns:
            Self for method chaining
        """
        for column, value in kwargs.items():
            if column not in self.data.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame")
            self.data = self.data[self.data[column] <= value]
        return self
    
    def isin(self, **kwargs) -> 'Transformable':
        """
        Filter rows where column values are in specified lists.
        
        Args:
            **kwargs: Column-list pairs for isin filtering
            
        Returns:
            Self for method chaining
            
        Example:
            .isin(category=['A', 'B', 'C'], status=['active', 'pending'])
        """
        for column, values in kwargs.items():
            if column not in self.data.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame")
            if not isinstance(values, (list, tuple, set, pd.Series)):
                raise ValueError(f"Value for column '{column}' must be a list, tuple, set, or Series")
            self.data = self.data[self.data[column].isin(values)]
        return self
    
    def between(self, **kwargs) -> 'Transformable':
        """
        Filter rows where column values are between specified ranges (inclusive).
        
        Args:
            **kwargs: Column-tuple pairs where tuple contains (min, max) values
            
        Returns:
            Self for method chaining
            
        Example:
            .between(price=(100, 200), age=(25, 65))
        """
        for column, range_values in kwargs.items():
            if column not in self.data.columns:
                raise KeyError(f"Column '{column}' not found in DataFrame")
            if not isinstance(range_values, (tuple, list)) or len(range_values) != 2:
                raise ValueError(f"Range for column '{column}' must be a tuple or list with exactly 2 values")
            
            min_val, max_val = range_values
            self.data = self.data[self.data[column].between(min_val, max_val, inclusive='both')]
        return self
    
    def reset_data(self, data: Optional[pd.DataFrame] = None) -> 'Transformable':
        """
        Reset the data to original or new DataFrame.
        
        Args:
            data: Optional new DataFrame to set. If None, reverts to original data.
            
        Returns:
            Self for method chaining
        """
        if data is not None:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("data must be a pandas DataFrame")
            self.data = data.copy()
        else:
            # Reset to original data - subclasses should override this method
            # to store and restore original data if needed
            raise NotImplementedError("Subclasses must implement reset functionality")
        return self
    
    def get_data(self) -> pd.DataFrame:
        """
        Get a copy of the current filtered DataFrame.
        
        Returns:
            Copy of the current data
        """
        return self.data.copy()
    
    def get_shape(self) -> tuple:
        """
        Get the shape of the current DataFrame.
        
        Returns:
            Tuple of (rows, columns)
        """
        return self.data.shape
    
    def get_columns(self) -> List[str]:
        """
        Get list of column names in the current DataFrame.
        
        Returns:
            List of column names
        """
        return list(self.data.columns)
    
    @abstractmethod
    def transform(self) -> pd.DataFrame:
        """
        Abstract method for custom transformation logic.
        
        Subclasses must implement this method to define their specific
        transformation behavior beyond basic filtering.
        
        Returns:
            Transformed DataFrame
        """
        return self.data
    
    def __len__(self) -> int:
        """Return the number of rows in the current DataFrame."""
        return len(self.data)
    
    def __repr__(self) -> str:
        """String representation showing class name and data shape."""
        return f"{self.__class__.__name__}(shape={self.data.shape})"