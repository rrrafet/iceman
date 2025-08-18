"""
Enhanced risk model definitions with temporal specifications and validation.

This module provides structured risk model definitions that include start_date,
end_date, and frequency specifications for better temporal control and validation.
"""

from datetime import datetime
from typing import List, Dict, Any, Callable, Optional
from pydantic import BaseModel, field_validator, Field, ConfigDict
import pandas as pd


class FactorSpecification(BaseModel):
    """Definition of a single factor in a risk model."""
    
    name: str = Field(..., description="Name of the factor")
    ticker: List[str] = Field(..., description="Bloomberg tickers for this factor")
    flds: List[str] = Field(..., description="Bloomberg fields to retrieve")
    func: Callable = Field(..., description="Transformation function to apply to raw data")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelMetadata(BaseModel):
    """Metadata for a risk model including temporal specifications."""
    
    start_date: str = Field(..., description="Start date for model data (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for model data (YYYY-MM-DD)")
    frequency: str = Field(..., description="Data frequency (e.g., 'B', 'D', 'W-MON')")
    description: Optional[str] = Field(None, description="Model description")
    version: str = Field("1.0", description="Model version")
    created_by: Optional[str] = Field(None, description="Model creator")
    created_date: Optional[str] = Field(None, description="Creation date")
    
    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v):
        """Validate date format is YYYY-MM-DD."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
    
    @field_validator('frequency')
    @classmethod
    def validate_frequency(cls, v):
        """Validate frequency against supported pandas frequencies."""
        from ..mappers import frequency_to_multiplier
        
        if v not in frequency_to_multiplier:
            supported = list(frequency_to_multiplier.keys())
            raise ValueError(f"Frequency '{v}' not supported. Use one of: {supported}")
        return v
    
    def validate_date_range(self):
        """Validate that start_date is before end_date."""
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        if start >= end:
            raise ValueError("start_date must be before end_date")


class RiskModelDefinition(BaseModel):
    """Complete risk model definition with metadata and factors."""
    
    metadata: ModelMetadata = Field(..., description="Model temporal and descriptive metadata")
    factors: List[FactorSpecification] = Field(..., description="List of factor specifications")
    
    def model_post_init(self, __context: Any):
        """Post-initialization validation."""
        self.metadata.validate_date_range()
    
    @property
    def factor_names(self) -> List[str]:
        """Get list of factor names."""
        return [factor.name for factor in self.factors]
    
    @property
    def all_tickers(self) -> List[str]:
        """Get all unique tickers used in this model."""
        tickers = set()
        for factor in self.factors:
            tickers.update(factor.ticker)
        return list(tickers)
    
    def get_factor_by_name(self, name: str) -> Optional[FactorSpecification]:
        """Get a factor specification by name."""
        for factor in self.factors:
            if factor.name == name:
                return factor
        return None
    
    def get_date_range(self) -> pd.DatetimeIndex:
        """Get pandas DatetimeIndex for the model's date range."""
        return pd.date_range(
            start=self.metadata.start_date,
            end=self.metadata.end_date,
            freq=self.metadata.frequency
        )
    


class RiskModelRegistry:
    """Registry for managing multiple risk models."""
    
    def __init__(self):
        self._models: Dict[str, RiskModelDefinition] = {}
    
    def register_model(self, name: str, model: RiskModelDefinition):
        """Register a new risk model."""
        if name in self._models:
            raise ValueError(f"Model '{name}' already exists. Use update_model() to modify.")
        self._models[name] = model
    
    def update_model(self, name: str, model: RiskModelDefinition):
        """Update an existing risk model."""
        self._models[name] = model
    
    def get_model(self, name: str) -> RiskModelDefinition:
        """Get a risk model by name."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available models: {list(self._models.keys())}")
        return self._models[name]
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get summary information about a model."""
        model = self.get_model(name)
        return {
            "name": name,
            "description": model.metadata.description,
            "start_date": model.metadata.start_date,
            "end_date": model.metadata.end_date,
            "frequency": model.metadata.frequency,
            "num_factors": len(model.factors),
            "factor_names": model.factor_names,
            "version": model.metadata.version
        }
    
    def remove_model(self, name: str):
        """Remove a model from the registry."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")
        del self._models[name]


# Global registry instance
model_registry = RiskModelRegistry()