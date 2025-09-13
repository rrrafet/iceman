import pandas as pd
from bbq import BBQ

def bloomberg_factor_returns(model: dict) -> pd.DataFrame:
    """
    Get factor returns from Bloomberg for a given risk model.
    :param model: The risk model from definitions.py to use (e.g., 'macro1', 'macro2').
    :param start_date: The start date for the data.
    :param end_date: The end date for the data.
    :param freq: The frequency of the data (e.g., 'B' for business days).
    :return: A DataFrame containing the factor returns.
    """
    start_date = model["start_date"]
    end_date = model["end_date"]
    freq = model["frequency"]
    factors = model["factors"]
    factor_returns = pd.DataFrame(index=pd.date_range(start_date, end_date, freq=freq))
    for component in factors:
        name, ticker, flds, func = component.values()
        descriptor = BBQ.bdh(tickers=ticker, flds=flds, start_date=start_date, end_date=end_date)
        if descriptor.empty:
            raise ValueError(f"Descriptor for {name} is empty. Please check the ticker and fields.")
        transformed = func(descriptor).iloc[:, 0]
        factor_returns = pd.concat([factor_returns, 
                             pd.Series(name=name, 
                                       data=transformed, 
                                       index=descriptor.index)], axis=1)
    return factor_returns

