# Initialize logging for risk module
try:
    from ...logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Unified risk decomposition architecture
from .decomposer import RiskDecomposer
from .context import (
    RiskContext, SingleModelContext, MultiModelContext, CustomContext, HierarchicalModelContext,
    create_single_model_context, create_active_risk_context, create_hierarchical_risk_context,
    create_portfolio_decomposer, create_active_risk_decomposer
)
from .calculator import RiskCalculator
from .strategies import RiskAnalysisStrategy, PortfolioAnalysisStrategy, ActiveRiskAnalysisStrategy, StrategyFactory

# Other risk module components
from .estimator import LinearRiskModel, LinearRiskModelEstimator
from .model import RiskModel
from .base import RiskDecomposerBase

__all__ = [
    # Unified interface
    'RiskDecomposer',
    'RiskContext', 'SingleModelContext', 'MultiModelContext', 'CustomContext', 'HierarchicalModelContext',
    'create_single_model_context', 'create_active_risk_context', 'create_hierarchical_risk_context',
    'create_portfolio_decomposer', 'create_active_risk_decomposer',
    'RiskCalculator',
    'RiskAnalysisStrategy', 'PortfolioAnalysisStrategy', 'ActiveRiskAnalysisStrategy', 'StrategyFactory',
    
    # Other components
    'LinearRiskModel',
    'LinearRiskModelEstimator',
    'RiskModel',
    'RiskDecomposerBase'
]