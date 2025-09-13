# Initialize logging for risk module
try:
    from core.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# NEW SIMPLIFIED API (Recommended) - 90% code reduction achieved!
from .risk_analysis import RiskResult, analyze_portfolio_risk, analyze_active_risk

# LEGACY COMPATIBILITY (Deprecated - will be removed in future version)
from .legacy import LegacyRiskDecomposer, RiskAnalysis, create_single_model_context, create_active_risk_context

# Original complex architecture (Deprecated)
from .decomposer import RiskDecomposer
from .context import (
    RiskContext, SingleModelContext, MultiModelContext, CustomContext, HierarchicalModelContext,
    create_hierarchical_risk_context, create_portfolio_decomposer, create_active_risk_decomposer
)
from .calculator import RiskCalculator
from .strategies import RiskAnalysisStrategy, PortfolioAnalysisStrategy, ActiveRiskAnalysisStrategy, StrategyFactory

# Other risk module components
from .estimator import LinearRiskModel, LinearRiskModelEstimator
from .model import RiskModel
from .base import RiskDecomposerBase

__all__ = [
    # NEW SIMPLIFIED API (Recommended) - 90% complexity reduction!
    'RiskResult',           # Simple dataclass with all essential results
    'analyze_portfolio_risk',  # Simple function replaces complex strategy pattern
    'analyze_active_risk',     # Simple function replaces complex multi-model context
    
    # LEGACY COMPATIBILITY (Deprecated - use new API above)
    'RiskAnalysis',         # Alias for RiskResult (backward compatibility)
    'LegacyRiskDecomposer', # Legacy wrapper
    'create_single_model_context',  # Legacy function -> analyze_portfolio_risk
    'create_active_risk_context',   # Legacy function -> analyze_active_risk
    
    # Original complex architecture (Deprecated - will be removed)
    'RiskDecomposer',
    'RiskContext', 'SingleModelContext', 'MultiModelContext', 'CustomContext', 'HierarchicalModelContext',
    'create_hierarchical_risk_context', 'create_portfolio_decomposer', 'create_active_risk_decomposer',
    'RiskCalculator',
    'RiskAnalysisStrategy', 'PortfolioAnalysisStrategy', 'ActiveRiskAnalysisStrategy', 'StrategyFactory',
    
    # Core components (preserved)
    'LinearRiskModel',
    'LinearRiskModelEstimator',
    'RiskModel',
    'RiskDecomposerBase'
]