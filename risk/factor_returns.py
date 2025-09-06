import pandas as pd
from typing import Union, List, Dict, Any, Optional
from spark.risk.model_definitions import RiskModelDefinition


def bloomberg_factor_returns(
    model: Union[RiskModelDefinition, List[Dict[str, Any]], str], 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    freq: Optional[str] = None
):
    """
    Get factor returns from Bloomberg for a given risk model.
    
    :param model: Risk model specification. Can be:
                 - RiskModelDefinition object (preferred)
                 - List of factor dictionaries (legacy format)
                 - String name of registered model
    :param start_date: Start date for the data (YYYY-MM-DD). If None and model is
                      RiskModelDefinition, uses model's start_date.
    :param end_date: End date for the data (YYYY-MM-DD). If None and model is 
                    RiskModelDefinition, uses model's end_date.
    :param freq: Frequency of the data (e.g., 'B' for business days). If None and 
                model is RiskModelDefinition, uses model's frequency.
    :return: A DataFrame containing the factor returns.
    """
    
    # Handle different input types
    if isinstance(model, str):
        # Model name - get from registry
        from spark.risk.definition import get_model
        model_def = get_model(model)
        factors_list = [
            {
                "name": factor.name,
                "ticker": factor.ticker,
                "flds": factor.flds,
                "func": factor.func
            }
            for factor in model_def.factors
        ]
        
        # Use model's temporal settings as defaults
        start_date = start_date or model_def.metadata.start_date
        end_date = end_date or model_def.metadata.end_date
        freq = freq or model_def.metadata.frequency
        
    elif isinstance(model, RiskModelDefinition):
        # Enhanced model definition
        model_def = model
        factors_list = [
            {
                "name": factor.name,
                "ticker": factor.ticker,
                "flds": factor.flds,
                "func": factor.func
            }
            for factor in model_def.factors
        ]
        
        # Use model's temporal settings as defaults
        start_date = start_date or model_def.metadata.start_date
        end_date = end_date or model_def.metadata.end_date
        freq = freq or model_def.metadata.frequency
        
    elif isinstance(model, list):
        # Legacy format - list of factor dictionaries
        factors_list = model
        model_def = None
        
        # All parameters must be provided for legacy format
        if not all([start_date, end_date, freq]):
            raise ValueError(
                "start_date, end_date, and freq must be provided when using legacy model format"
            )
    else:
        raise TypeError(
            f"model must be RiskModelDefinition, list of dicts, or string, got {type(model)}"
        )

    # Validate required parameters
    if not all([start_date, end_date, freq]):
        raise ValueError("start_date, end_date, and freq are required")

    # Create empty DataFrame with date index
    factors = pd.DataFrame(index=pd.date_range(start_date, end_date, freq=freq))
    
    # Process each factor
    for component in factors_list:
        name = component["name"]
        ticker = component["ticker"] 
        flds = component["flds"]
        func = component["func"]
        
        # Get data from Bloomberg (assuming bloomberg function exists)
        descriptor = bloomberg(tickers=ticker, flds=flds, start_date=start_date, end_date=end_date)
        
        if descriptor.empty:
            raise ValueError(f"Descriptor for {name} is empty. Please check the ticker and fields.")
        
        # Apply transformation function
        transformed = func(descriptor).iloc[:, 0]
        
        # Add to factors DataFrame
        factors = pd.concat([
            factors, 
            pd.Series(name=name, data=transformed, index=descriptor.index)
        ], axis=1)
    
    return factors




# Note: The bloomberg() function is assumed to exist - it should be imported from 
# the appropriate Bloomberg API module or implemented separately

