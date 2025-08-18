"""
Risk analysis strategies for different types of risk decomposition.

This module implements the strategy pattern for risk analysis, allowing
different decomposition approaches to be plugged into the unified decomposer.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, TYPE_CHECKING
from .calculator import RiskCalculator
from .annualizer import RiskAnnualizer
from .context import RiskContext, SingleModelContext, MultiModelContext

if TYPE_CHECKING:
    from .schema import RiskResultSchema


class RiskAnalysisStrategy(ABC):
    """
    Abstract base class for risk analysis strategies.
    
    Each strategy implements a specific approach to risk decomposition,
    such as traditional portfolio analysis or active risk analysis.
    """
    
    @abstractmethod
    def analyze(self, context: RiskContext) -> 'RiskResultSchema':
        """
        Perform risk analysis given a context.
        
        Parameters
        ----------
        context : RiskContext
            Risk analysis context containing models and weights
            
        Returns
        -------
        RiskResultSchema
            Risk analysis results in unified schema format
        """
        pass
    
    @abstractmethod
    def get_supported_context_types(self) -> Tuple[type, ...]:
        """Return tuple of supported context types"""
        pass
    


class PortfolioAnalysisStrategy(RiskAnalysisStrategy):
    """
    Strategy for traditional single-portfolio risk analysis.
    
    This strategy performs standard portfolio risk decomposition into
    factor and specific (idiosyncratic) components using the unified schema.
    """
    
    def analyze(self, context: RiskContext) -> 'RiskResultSchema':
        """
        Core portfolio risk analysis returning unified schema results.
        
        Parameters
        ----------
        context : SingleModelContext
            Single-model risk context
            
        Returns
        -------
        RiskResultSchema
            Portfolio risk analysis results in unified schema format
        """
        if not isinstance(context, SingleModelContext):
            raise TypeError(f"PortfolioAnalysisStrategy requires SingleModelContext, got {type(context)}")
        
        model = context.model
        weights = context.weights
        annualize = context.annualize
        frequency = context.frequency
        
        # Create unified schema
        from .schema import RiskResultSchema, AnalysisType
        schema = RiskResultSchema(
            analysis_type=AnalysisType.PORTFOLIO,
            asset_names=context.get_asset_names(),
            factor_names=context.get_factor_names(),
            data_frequency=frequency,
            annualized=annualize
        )
        
        # Core risk calculations (all raw, non-annualized)
        portfolio_volatility = RiskCalculator.calculate_portfolio_volatility(
            model.covar, weights
        )
        
        # Factor exposures and contributions
        factor_exposures = RiskCalculator.calculate_factor_exposures(model.beta, weights)
        factor_contributions = RiskCalculator.calculate_factor_contributions(
            model.beta, model.factor_covar, weights, portfolio_volatility
        )
        marginal_factor_contributions = RiskCalculator.calculate_marginal_factor_contributions(
            model.beta, model.factor_covar, weights, portfolio_volatility
        )
        
        # Specific risk contributions
        specific_contributions = RiskCalculator.calculate_specific_contributions(
            model.resvar, weights, portfolio_volatility
        )
        
        # Asset-level contributions
        marginal_contributions = RiskCalculator.calculate_marginal_contributions(
            model.covar, weights, portfolio_volatility
        )
        
        asset_contributions = RiskCalculator.calculate_total_contributions(
            marginal_contributions, weights, portfolio_volatility
        )
        
        # Factor and specific asset-level breakdowns
        asset_factor_contributions = RiskCalculator.calculate_asset_factor_contributions(
            model.beta, model.factor_covar, weights, portfolio_volatility
        )
        
        asset_specific_contributions = RiskCalculator.calculate_asset_specific_contributions(
            model.resvar, weights, portfolio_volatility
        )
        
        # Asset by factor contributions matrix (non-standard, detailed breakdown)
        asset_by_factor_contributions = RiskCalculator.calculate_asset_by_factor_contributions(
            model.beta, model.factor_covar, weights, portfolio_volatility
        )
        
        # Weighted betas matrix 
        weighted_betas = RiskCalculator.calculate_weighted_betas(
            model.beta, weights
        )
        
        # Percentage contributions
        percent_contributions = RiskCalculator.calculate_percent_contributions(
            asset_contributions, portfolio_volatility
        )
        percent_factor_contributions = RiskCalculator.calculate_percent_factor_contributions(
            factor_contributions, portfolio_volatility
        )
        
        # Validation
        validation = RiskCalculator.validate_risk_decomposition(
            portfolio_volatility,
            {
                'factor_risk_contribution': factor_contributions.sum(),
                'specific_risk_contribution': specific_contributions.sum()
            }
        )
        
        # Set core metrics (apply annualization if requested)
        if annualize:
            annualized_results = RiskAnnualizer.annualize_risk_results({
                'portfolio_volatility': portfolio_volatility,
                'factor_risk_contribution': factor_contributions.sum(),
                'specific_risk_contribution': specific_contributions.sum()
            }, frequency)
            schema.set_core_metrics(
                annualized_results['portfolio_volatility'],
                annualized_results['factor_risk_contribution'],
                annualized_results['specific_risk_contribution']
            )
        else:
            schema.set_core_metrics(
                portfolio_volatility,
                factor_contributions.sum(),
                specific_contributions.sum()
            )
        
        # Set exposures
        schema.set_factor_exposures(factor_exposures)
        
        # Set contributions (annualize if requested)
        if annualize:
            annualized_asset_contrib = RiskAnnualizer.annualize_contributions(asset_contributions, frequency)
            annualized_factor_contrib = RiskAnnualizer.annualize_contributions(factor_contributions, frequency)
            schema.set_asset_contributions(annualized_asset_contrib)
            schema.set_factor_contributions(annualized_factor_contrib)
        else:
            schema.set_asset_contributions(asset_contributions)
            schema.set_factor_contributions(factor_contributions)
        
        # Set matrix data
        schema.set_factor_risk_contributions_matrix(asset_by_factor_contributions)
        schema.set_weighted_betas_matrix(weighted_betas)
        
        # Set validation results
        schema.set_validation_results(validation)
        
        # Add detailed analysis information
        schema.add_detail('marginal_factor_contributions', marginal_factor_contributions.tolist())
        schema.add_detail('asset_factor_contributions', asset_factor_contributions.tolist())
        schema.add_detail('asset_specific_contributions', asset_specific_contributions.tolist())
        schema.add_detail('asset_by_factor_contributions', asset_by_factor_contributions.tolist())
        schema.add_detail('marginal_asset_contributions', marginal_contributions.tolist())
        schema.add_detail('percent_total_contributions', percent_contributions.tolist())
        schema.add_detail('percent_factor_contributions', percent_factor_contributions.tolist())
        
        # Add context information
        schema.add_context_info('model_type', type(model).__name__)
        schema.add_context_info('calculation_method', 'RiskCalculator')
        
        return schema
    
    def get_supported_context_types(self) -> Tuple[type, ...]:
        return (SingleModelContext,)


class ActiveRiskAnalysisStrategy(RiskAnalysisStrategy):
    """
    Strategy for active risk analysis using Brinson-style decomposition.
    
    This strategy decomposes active risk into allocation and selection components,
    each further broken down into factor and specific risk using the unified schema.
    """
    
    def analyze(self, context: RiskContext) -> 'RiskResultSchema':
        """
        Core active risk analysis returning unified schema results.
        
        Parameters
        ----------
        context : MultiModelContext
            Multi-model risk context
            
        Returns
        -------
        RiskResultSchema
            Active risk analysis results in unified schema format
        """
        if not isinstance(context, MultiModelContext):
            raise TypeError(f"ActiveRiskAnalysisStrategy requires MultiModelContext, got {type(context)}")
        
        # Extract models and weights
        portfolio_model = context.portfolio_model
        benchmark_model = context.benchmark_model
        active_model = context.active_model
        portfolio_weights = context.portfolio_weights
        benchmark_weights = context.benchmark_weights
        active_weights = context.active_weights
        annualize = context.annualize
        frequency = context.frequency
        
        # Calculate total active risk (raw) including cross-correlation
        # Check if cross-covariance is available in the context
        cross_covar = getattr(context, 'cross_covar', None)
        
        total_active_risk = RiskCalculator.calculate_active_risk(
            benchmark_model.covar,
            portfolio_weights, benchmark_weights,
            active_model.covar,
            cross_covar
        )
        
        # Factor exposures
        portfolio_factor_exposure = RiskCalculator.calculate_factor_exposures(
            portfolio_model.beta, portfolio_weights
        )
        benchmark_factor_exposure = RiskCalculator.calculate_factor_exposures(
            benchmark_model.beta, benchmark_weights
        )
        active_factor_exposure = portfolio_factor_exposure - benchmark_factor_exposure
        
        # Allocation components (using benchmark model)
        allocation_factor_exposure = RiskCalculator.calculate_factor_exposures(
            benchmark_model.beta, active_weights
        )
        
        allocation_factor_risk = RiskCalculator.calculate_factor_risk(
            benchmark_model.beta, benchmark_model.factor_covar,
            active_weights
        )
        
        allocation_specific_risk = RiskCalculator.calculate_specific_risk(
            benchmark_model.resvar, active_weights
        )
        
        total_allocation_risk = np.sqrt(
            (allocation_factor_risk ** 2) + (allocation_specific_risk ** 2)
        )
        
        # Selection components (using differences in factor loadings)
        beta_diff = portfolio_model.beta - benchmark_model.beta
        selection_factor_exposure = RiskCalculator.calculate_factor_exposures(beta_diff, portfolio_weights)
        
        selection_factor_risk = RiskCalculator.calculate_factor_risk(
            beta_diff, active_model.factor_covar,
            portfolio_weights
        )
        
        # Selection specific risk (difference in specific characteristics)
        selection_specific_risk = RiskCalculator.calculate_specific_risk(
            active_model.resvar, portfolio_weights
        )
        
        total_selection_risk = np.sqrt(
            (selection_factor_risk ** 2) + (selection_specific_risk ** 2)
        )
        
        # Cross-correlation components (if available)
        cross_risk_contribution = 0.0
        cross_euler_contribution = 0.0
        cross_variance_contribution = 0.0
        
        if cross_covar is not None:
            # Calculate cross-variance contribution: 2 d^T C w
            cross_variance_contribution = RiskCalculator.calculate_cross_variance_contribution(
                cross_covar, active_weights, portfolio_weights
            )
            
            # Calculate Euler cross-risk contribution
            if total_active_risk > 0:
                cross_euler_contribution = RiskCalculator.calculate_euler_cross_contributions(
                    cross_covar, active_weights, portfolio_weights, total_active_risk
                )
                cross_risk_contribution = np.sqrt(max(0.0, cross_variance_contribution)) if cross_variance_contribution >= 0 else -np.sqrt(abs(cross_variance_contribution))
        
        # Asset-level contributions
        if total_active_risk == 0:
            n_assets = len(portfolio_weights)
            asset_contributions = {
                'total': np.zeros(n_assets),
                'allocation_factor': np.zeros(n_assets),
                'allocation_specific': np.zeros(n_assets),
                'selection_factor': np.zeros(n_assets),
                'selection_specific': np.zeros(n_assets),
                'cross_correlation': np.zeros(n_assets)
            }
        else:
            # Calculate asset contributions using RiskCalculator (raw values)
            allocation_factor = RiskCalculator.calculate_asset_allocation_factor_contributions(
                benchmark_model.beta, benchmark_model.factor_covar, active_weights,
                total_active_risk
            )
            
            allocation_specific = RiskCalculator.calculate_asset_allocation_specific_contributions(
                benchmark_model.resvar, active_weights,
                total_active_risk
            )
            
            selection_factor = RiskCalculator.calculate_asset_selection_factor_contributions(
                portfolio_model.beta, benchmark_model.beta, active_model.factor_covar,
                portfolio_weights, total_active_risk
            )
            
            selection_specific = RiskCalculator.calculate_asset_selection_specific_contributions(
                active_model.resvar, portfolio_weights,
                total_active_risk
            )
            
            # Cross-correlation asset contributions (enhanced decomposition)
            cross_asset_contributions = np.zeros(len(portfolio_weights))
            cross_factor_contributions = np.zeros(len(portfolio_weights))
            cross_specific_contributions = np.zeros(len(portfolio_weights))
            
            if cross_covar is not None:
                # Total cross-correlation contributions
                cross_asset_contributions = RiskCalculator.calculate_asset_level_cross_contributions(
                    cross_covar, active_weights, portfolio_weights, total_active_risk
                )
                
                # Decompose cross-correlation into factor and specific components
                try:
                    cross_factor_contributions = RiskCalculator.calculate_asset_cross_correlation_factor_contributions(
                        cross_covar, benchmark_model.beta, active_model.beta,
                        active_weights, portfolio_weights, total_active_risk
                    )
                    # Specific cross-correlation = total cross - factor cross
                    cross_specific_contributions = cross_asset_contributions - cross_factor_contributions
                except Exception:
                    # Fallback to simple decomposition if enhanced method fails
                    cross_factor_contributions = cross_asset_contributions * 0.7  # Rough approximation
                    cross_specific_contributions = cross_asset_contributions * 0.3
            
            asset_contributions = {
                'total': allocation_factor + allocation_specific + selection_factor + selection_specific + cross_asset_contributions,
                'allocation_factor': allocation_factor,
                'allocation_specific': allocation_specific,
                'selection_factor': selection_factor,
                'selection_specific': selection_specific,
                'cross_correlation': cross_asset_contributions,
                'cross_correlation_factor': cross_factor_contributions,
                'cross_correlation_specific': cross_specific_contributions
            }
        
        # Factor-level contributions (enhanced with cross-correlation)
        if total_active_risk == 0:
            n_factors = portfolio_model.beta.shape[1]
            factor_contributions = {
                'total': np.zeros(n_factors),
                'allocation': np.zeros(n_factors),
                'selection': np.zeros(n_factors),
                'cross_correlation': np.zeros(n_factors)
            }
        else:
            # Allocation factor contributions (raw)
            allocation_exposure = RiskCalculator.calculate_factor_exposures(
                benchmark_model.beta, active_weights
            )
            allocation_marginal = (benchmark_model.factor_covar @ allocation_exposure) / total_active_risk
            allocation_contributions = allocation_exposure * allocation_marginal
            
            # Selection factor contributions (raw) 
            beta_diff = portfolio_model.beta - benchmark_model.beta
            selection_exposure = RiskCalculator.calculate_factor_exposures(beta_diff, portfolio_weights)
            selection_marginal = (active_model.factor_covar @ selection_exposure) / total_active_risk
            selection_contributions = selection_exposure * selection_marginal
            
            # Cross-correlation factor contributions
            n_factors = portfolio_model.beta.shape[1]
            cross_factor_contributions = np.zeros(n_factors)
            if cross_covar is not None:
                try:
                    cross_factor_contributions = RiskCalculator.calculate_factor_level_cross_contributions(
                        cross_covar, benchmark_model.beta, active_model.beta,
                        active_weights, portfolio_weights, total_active_risk
                    )
                except Exception:
                    # Fallback if calculation fails
                    pass
            
            factor_contributions = {
                'total': allocation_contributions + selection_contributions + cross_factor_contributions,
                'allocation': allocation_contributions,
                'selection': selection_contributions,
                'cross_correlation': cross_factor_contributions
            }
        
        # Risk decomposition percentages
        if total_active_risk == 0:
            risk_decomp = {
                'allocation_factor': 0.0,
                'allocation_specific': 0.0,
                'selection_factor': 0.0,
                'selection_specific': 0.0,
                'cross_correlation': 0.0,
                'total_allocation': 0.0,
                'total_selection': 0.0,
                'total_factor': 0.0,
                'total_specific': 0.0
            }
        else:
            # Calculate as percentage of total variance (not volatility)
            total_var = total_active_risk ** 2
            
            # Use RiskCalculator for consistent percentage calculations
            component_variances = np.array([
                allocation_factor_risk ** 2,
                allocation_specific_risk ** 2,
                selection_factor_risk ** 2,
                selection_specific_risk ** 2,
                cross_variance_contribution  # This is already variance, not volatility
            ])
            
            component_percentages = RiskCalculator.calculate_percent_contributions(
                component_variances, total_var
            )
            
            alloc_factor_pct, alloc_specific_pct, sel_factor_pct, sel_specific_pct, cross_pct = component_percentages
            
            risk_decomp = {
                'allocation_factor': alloc_factor_pct,
                'allocation_specific': alloc_specific_pct,
                'selection_factor': sel_factor_pct,
                'selection_specific': sel_specific_pct,
                'cross_correlation': cross_pct,
                'total_allocation': alloc_factor_pct + alloc_specific_pct,
                'total_selection': sel_factor_pct + sel_specific_pct,
                'total_factor': alloc_factor_pct + sel_factor_pct,
                'total_specific': alloc_specific_pct + sel_specific_pct
            }
        
        # Enhanced Euler identity validation including cross-correlation
        component_risks = {
            'allocation_factor': allocation_factor_risk,
            'allocation_specific': allocation_specific_risk,
            'selection_factor': selection_factor_risk,
            'selection_specific': selection_specific_risk
        }
        
        # Add cross-correlation if present
        if cross_covar is not None:
            component_risks['cross_correlation'] = cross_euler_contribution
        
        # Use enhanced validation methods
        validation = RiskCalculator.validate_risk_decomposition(
            total_active_risk, component_risks, tolerance=1e-6, include_cross_correlation=True
        )
        
        # Additional specialized validation for active risk
        active_risk_validation = RiskCalculator.validate_active_risk_euler_identity(
            total_active_risk, 
            allocation_factor_risk, allocation_specific_risk,
            selection_factor_risk, selection_specific_risk,
            cross_euler_contribution, tolerance=1e-6
        )
        
        # Combined interface properties for compatibility
        factor_risk_contribution = allocation_factor_risk + selection_factor_risk
        specific_risk_contribution = total_active_risk - factor_risk_contribution
        
        # Create unified schema
        from .schema import RiskResultSchema, AnalysisType
        schema = RiskResultSchema(
            analysis_type=AnalysisType.ACTIVE,
            asset_names=context.get_asset_names(),
            factor_names=context.get_factor_names(),
            data_frequency=frequency,
            annualized=annualize
        )
        
        # Set core metrics (apply annualization if requested)
        if annualize:
            annualized_results = RiskAnnualizer.annualize_risk_results({
                'portfolio_volatility': total_active_risk,
                'factor_risk_contribution': factor_risk_contribution,
                'specific_risk_contribution': specific_risk_contribution
            }, frequency)
            schema.set_core_metrics(
                annualized_results['portfolio_volatility'],
                annualized_results['factor_risk_contribution'],
                annualized_results['specific_risk_contribution']
            )
        else:
            schema.set_core_metrics(
                total_active_risk,
                factor_risk_contribution,
                specific_risk_contribution
            )
        
        # Set exposures
        schema.set_factor_exposures(active_factor_exposure)
        
        # Set contributions (annualize if requested)
        if annualize:
            annualized_asset_contrib = RiskAnnualizer.annualize_contributions(asset_contributions['total'], frequency)
            annualized_factor_contrib = RiskAnnualizer.annualize_contributions(factor_contributions['total'], frequency)
            schema.set_asset_contributions(annualized_asset_contrib)
            schema.set_factor_contributions(annualized_factor_contrib)
        else:
            schema.set_asset_contributions(asset_contributions['total'])
            schema.set_factor_contributions(factor_contributions['total'])
        
        # Calculate matrix data for active risk (use portfolio model as primary)
        portfolio_volatility = RiskCalculator.calculate_portfolio_volatility(
            portfolio_model.covar, portfolio_weights
        )
        
        # Asset by factor contributions matrix using portfolio model
        portfolio_asset_by_factor_contributions = RiskCalculator.calculate_asset_by_factor_contributions(
            portfolio_model.beta, portfolio_model.factor_covar, portfolio_weights, portfolio_volatility
        )
        
        # Weighted betas matrix for portfolio
        portfolio_weighted_betas = RiskCalculator.calculate_weighted_betas(
            portfolio_model.beta, portfolio_weights
        )
        
        # Set matrix data
        schema.set_factor_risk_contributions_matrix(portfolio_asset_by_factor_contributions)
        schema.set_weighted_betas_matrix(portfolio_weighted_betas)
        
        # Set validation results (combine both validation types)
        combined_validation = validation.copy()
        combined_validation['active_risk_validation'] = active_risk_validation
        schema.set_validation_results(combined_validation)
        
        # Set active risk specific metrics
        active_risk_metrics = {
            'total_active_risk': total_active_risk,
            'allocation_factor_risk': allocation_factor_risk,
            'allocation_specific_risk': allocation_specific_risk,
            'selection_factor_risk': selection_factor_risk,
            'selection_specific_risk': selection_specific_risk,
            'cross_correlation_risk': cross_risk_contribution,
            'cross_euler_contribution': cross_euler_contribution,
            'cross_variance_contribution': cross_variance_contribution,
            'total_allocation_risk': total_allocation_risk,
            'total_selection_risk': total_selection_risk,
            'risk_decomposition': risk_decomp
        }
        schema.set_active_risk_metrics(active_risk_metrics)
        
        # Add detailed analysis information
        schema.add_detail('portfolio_factor_exposure', portfolio_factor_exposure.tolist())
        schema.add_detail('benchmark_factor_exposure', benchmark_factor_exposure.tolist())
        schema.add_detail('allocation_factor_exposure', allocation_factor_exposure.tolist())
        schema.add_detail('selection_factor_exposure', selection_factor_exposure.tolist())
        
        schema.add_detail('asset_allocation_factor_contributions', asset_contributions['allocation_factor'].tolist())
        schema.add_detail('asset_allocation_specific_contributions', asset_contributions['allocation_specific'].tolist())
        schema.add_detail('asset_selection_factor_contributions', asset_contributions['selection_factor'].tolist())
        schema.add_detail('asset_selection_specific_contributions', asset_contributions['selection_specific'].tolist())
        schema.add_detail('asset_cross_correlation_contributions', asset_contributions['cross_correlation'].tolist())
        schema.add_detail('asset_cross_correlation_factor_contributions', asset_contributions['cross_correlation_factor'].tolist())
        schema.add_detail('asset_cross_correlation_specific_contributions', asset_contributions['cross_correlation_specific'].tolist())
        
        schema.add_detail('factor_allocation_contributions', factor_contributions['allocation'].tolist())
        schema.add_detail('factor_selection_contributions', factor_contributions['selection'].tolist())
        schema.add_detail('factor_cross_correlation_contributions', factor_contributions['cross_correlation'].tolist())
        
        # Add context information
        schema.add_context_info('portfolio_model_type', type(portfolio_model).__name__)
        schema.add_context_info('benchmark_model_type', type(benchmark_model).__name__)
        schema.add_context_info('active_model_type', type(active_model).__name__)
        schema.add_context_info('calculation_method', 'RiskCalculator')
        schema.add_context_info('cross_correlation_included', cross_covar is not None)
        
        return schema
    
    def get_supported_context_types(self) -> Tuple[type, ...]:
        return (MultiModelContext,)
    


class StrategyFactory:
    """Factory for creating appropriate risk analysis strategies"""
    
    _strategies = {
        'portfolio': PortfolioAnalysisStrategy,
        'active': ActiveRiskAnalysisStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_type: str) -> RiskAnalysisStrategy:
        """
        Create a risk analysis strategy.
        
        Parameters
        ----------
        strategy_type : str
            Type of strategy ('portfolio' or 'active')
            
        Returns
        -------
        RiskAnalysisStrategy
            Strategy instance
        """
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}. Available: {list(cls._strategies.keys())}")
        
        return cls._strategies[strategy_type]()
    
    @classmethod
    def get_strategy_for_context(cls, context: RiskContext) -> RiskAnalysisStrategy:
        """
        Get appropriate strategy for a given context.
        
        Parameters
        ----------
        context : RiskContext
            Risk analysis context
            
        Returns
        -------
        RiskAnalysisStrategy
            Appropriate strategy for the context
        """
        if isinstance(context, SingleModelContext):
            return cls.create_strategy('portfolio')
        elif isinstance(context, MultiModelContext):
            return cls.create_strategy('active')
        else:
            raise ValueError(f"No default strategy available for context type: {type(context)}")
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """Register a custom strategy"""
        cls._strategies[name] = strategy_class