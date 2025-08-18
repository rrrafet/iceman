"""
Risk model definitions using enhanced structure with temporal specifications.

This file defines risk models with explicit start_date, end_date, and frequency
specifications for better temporal control and validation.
"""

from spark.risk.model_definitions import (
    RiskModelDefinition, 
    ModelMetadata, 
    FactorSpecification,
    model_registry
)


# Define the enhanced macro1 model
macro1_model = RiskModelDefinition(
    metadata=ModelMetadata(
        start_date="2020-01-01",
        end_date="2024-12-31",
        frequency="B",  # Business daily
        description="Basic macro factor model with market, bonds, dollar, commodity, and credit factors",
        version="1.0",
        created_by="Risk Team"
    ),
    factors=[
        FactorSpecification(
            name="Market",
            ticker=["MXWO Index"],
            flds=["LAST_PRICE"],
            func=lambda x: x.pct_change(),
        ),
        FactorSpecification(
            name="Bonds",
            ticker=["USGG10YR Index"],
            flds=["LAST_PRICE"],
            func=lambda x: 0.01 * (x["USGG10YR Index"]).diff(),
        ),
        FactorSpecification(
            name="Dollar",
            ticker=["DXY Index"],
            flds=["LAST_PRICE"],
            func=lambda x: x.pct_change(),
        ),
        FactorSpecification(
            name="Commodity",
            ticker=["SPGSCI Index"],
            flds=["LAST_PRICE"],
            func=lambda x: x.pct_change(),
        ),
        FactorSpecification(
            name="Credit",
            ticker=["LF98OAS Index"],
            flds=["LAST_PRICE"],
            func=lambda x: x.pct_change(),
        ),
    ]
)

# Define the enhanced macro2 model  
macro2_model = RiskModelDefinition(
    metadata=ModelMetadata(
        start_date="2020-01-01",
        end_date="2024-12-31", 
        frequency="B",  # Business daily
        description="Enhanced macro factor model with yield curve and AI factors",
        version="1.1",
        created_by="Risk Team"
    ),
    factors=[
        FactorSpecification(
            name="Market",
            ticker=["MXWO Index"],
            flds=["LAST_PRICE"],
            func=lambda x: x.pct_change(),
        ),
        FactorSpecification(
            name="Curve",
            ticker=["USGG10YR Index", "USGG2YR Index"],
            flds=["LAST_PRICE"],
            func=lambda x: 0.01 * (x["USGG10YR Index"] - x["USGG2YR Index"]).diff(),
        ),
        FactorSpecification(
            name="Dollar",
            ticker=["DXY Index"],
            flds=["LAST_PRICE"],
            func=lambda x: x.pct_change(),
        ),
        FactorSpecification(
            name="Commodity",
            ticker=["SPGSCI Index"],
            flds=["LAST_PRICE"],
            func=lambda x: x.pct_change(),
        ),
        FactorSpecification(
            name="Credit",
            ticker=["LF98OAS Index"],
            flds=["LAST_PRICE"],
            func=lambda x: x.pct_change(),
        ),
        FactorSpecification(
            name="AI",
            ticker=["MSXXAIB Index", "SPX Index"],
            flds=["CHG_PCT_1D"],
            func=lambda x: 0.01 * (x["MSXXAIB Index"] - 1.5 * x["SPX Index"]),
        ),
    ]
)

# Register models in the global registry
model_registry.register_model("macro1", macro1_model)
model_registry.register_model("macro2", macro2_model)

# Legacy dictionary format removed - use get_model() instead

# Convenience functions for accessing models
def get_model(name: str) -> RiskModelDefinition:
    """Get a risk model by name."""
    return model_registry.get_model(name)

def list_available_models():
    """List all available risk models."""
    return model_registry.list_models()

def get_model_info(name: str):
    """Get information about a specific model."""
    return model_registry.get_model_info(name)
