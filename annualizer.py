"""
Risk annualization utilities.

This module provides centralized annualization logic, separating mathematical
calculations from presentation formatting. All core risk calculations should
return raw (non-annualized) values, and this utility handles conversion to
annualized form when needed for reporting.
"""

import numpy as np
from typing import Dict, Any
from spark.core.mappers import frequency_to_multiplier


class RiskAnnualizer:
    """
    Centralized utility for annualizing risk metrics.
    
    This class provides methods to convert raw risk calculations to annualized
    form for reporting and presentation purposes. It consolidates all 
    annualization logic in one place to eliminate code duplication.
    """
    
    @staticmethod
    def annualize_volatility(
        volatility, 
        frequency: str = "D"
    ):
        """
        Convert raw volatility to annualized form.
        
        Parameters
        ----------
        volatility : float, numpy.ndarray, or pandas.Series
            Raw (non-annualized) volatility
        frequency : str, default "D"
            Data frequency for annualization
            
        Returns
        -------
        float, numpy.ndarray, or pandas.Series
            Annualized volatility (same type as input)
        """
        # Handle zero values
        if hasattr(volatility, '__iter__'):
            # For arrays/Series, use vectorized operations
            multiplier = frequency_to_multiplier.get(frequency.upper(), 1.0)
            return volatility * np.sqrt(multiplier)
        else:
            # For scalars
            if volatility == 0:
                return 0.0
            multiplier = frequency_to_multiplier.get(frequency.upper(), 1.0)
            return volatility * np.sqrt(multiplier)
    
    @staticmethod
    def annualize_contributions(
        contributions: np.ndarray,
        frequency: str = "D"
    ) -> np.ndarray:
        """
        Convert raw contributions to annualized form.
        
        Parameters
        ----------
        contributions : np.ndarray
            Raw (non-annualized) contributions
        frequency : str, default "D"
            Data frequency for annualization
            
        Returns
        -------
        np.ndarray
            Annualized contributions
        """
        contributions = np.asarray(contributions)
        if contributions.size == 0 or np.all(contributions == 0):
            return contributions
            
        multiplier = frequency_to_multiplier.get(frequency.upper(), 1.0)
        return contributions * np.sqrt(multiplier)
    
    @staticmethod
    def annualize_risk_results(
        results: Dict[str, Any],
        frequency: str = "D"
    ) -> Dict[str, Any]:
        """
        Annualize all risk metrics in a results dictionary.
        
        This method identifies risk-related keys in the results dictionary
        and applies appropriate annualization. It modifies the dictionary
        in-place and returns it for convenience.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Results dictionary containing risk metrics
        frequency : str, default "D"
            Data frequency for annualization
            
        Returns
        -------
        Dict[str, Any]
            Results dictionary with annualized values
        """
        if not results:
            return results
        
        # Keys that represent volatilities/risks (scalars)
        volatility_keys = {
            'portfolio_volatility',
            'factor_risk',
            'total_risk',
            'active_risk',
            'specific_risk',
            'total_active_risk',
            'allocation_factor_risk',
            'allocation_specific_risk', 
            'selection_factor_risk',
            'selection_specific_risk',
            'total_allocation_risk',
            'total_selection_risk',
            'factor_risk_contribution',
            'specific_risk_contribution'
        }
        
        # Keys that represent contributions (arrays)
        contribution_keys = {
            'factor_contributions',
            'marginal_factor_contributions',
            'asset_total_contributions',
            'asset_factor_contributions',
            'asset_specific_contributions',
            'marginal_asset_contributions',
            'asset_allocation_factor_contributions',
            'asset_allocation_specific_contributions',
            'asset_selection_factor_contributions', 
            'asset_selection_specific_contributions',
            'factor_allocation_contributions',
            'factor_selection_contributions'
        }
        
        # Keys that are matrices (2D arrays)
        matrix_keys = {
            'asset_by_factor_contributions'
        }
        
        # Annualize scalar volatilities
        for key in volatility_keys:
            if key in results and isinstance(results[key], (int, float)):
                results[key] = RiskAnnualizer.annualize_volatility(
                    results[key], frequency
                )
        
        # Annualize contribution arrays
        for key in contribution_keys:
            if key in results and isinstance(results[key], np.ndarray):
                results[key] = RiskAnnualizer.annualize_contributions(
                    results[key], frequency
                )
        
        # Annualize contribution matrices
        for key in matrix_keys:
            if key in results and isinstance(results[key], np.ndarray):
                results[key] = RiskAnnualizer.annualize_contributions(
                    results[key], frequency
                )
        
        # Set annualization metadata
        results['annualized'] = True
        results['frequency'] = frequency
        
        return results
    
    @staticmethod
    def de_annualize_volatility(
        annualized_volatility: float,
        frequency: str = "D"
    ) -> float:
        """
        Convert annualized volatility back to raw form.
        
        This is useful when working with pre-annualized inputs that need
        to be converted back to raw form for calculations.
        
        Parameters
        ----------
        annualized_volatility : float
            Annualized volatility
        frequency : str, default "D"
            Data frequency used for original annualization
            
        Returns
        -------
        float
            Raw (non-annualized) volatility
        """
        if annualized_volatility == 0:
            return 0.0
            
        multiplier = frequency_to_multiplier.get(frequency.upper(), 1.0)
        return annualized_volatility / np.sqrt(multiplier)
    
    @staticmethod
    def get_annualization_multiplier(frequency: str = "D") -> float:
        """
        Get the annualization multiplier for a given frequency.
        
        Parameters
        ----------
        frequency : str, default "D"
            Data frequency
            
        Returns
        -------
        float
            Annualization multiplier (square root of frequency multiplier)
        """
        multiplier = frequency_to_multiplier.get(frequency.upper(), 1.0)
        return np.sqrt(multiplier)
    
    @staticmethod
    def annualize_return(
        mean_return: float,
        frequency: str = "D"
    ) -> float:
        """
        Convert raw mean return to annualized form.
        
        Parameters
        ----------
        mean_return : float
            Raw (non-annualized) mean return
        frequency : str, default "D"
            Data frequency for annualization
            
        Returns
        -------
        float
            Annualized return
        """
        multiplier = frequency_to_multiplier.get(frequency.upper(), 1.0)
        return mean_return * multiplier
    
    @staticmethod
    def de_annualize_return(
        annualized_return: float,
        frequency: str = "D"
    ) -> float:
        """
        Convert annualized return back to raw form.
        
        Parameters
        ----------
        annualized_return : float
            Annualized return
        frequency : str, default "D"
            Data frequency used for original annualization
            
        Returns
        -------
        float
            Raw (non-annualized) return
        """
        multiplier = frequency_to_multiplier.get(frequency.upper(), 1.0)
        return annualized_return / multiplier
    
    @staticmethod
    def get_periods_per_year(frequency: str = "D") -> float:
        """
        Get the number of periods per year for a given frequency.
        
        This is useful for display formatting and time period calculations.
        
        Parameters
        ----------
        frequency : str, default "D"
            Data frequency
            
        Returns
        -------
        float
            Number of periods per year
        """
        return frequency_to_multiplier.get(frequency.upper(), 1.0)