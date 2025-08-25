"""
Services package for Maverick risk analysis application.

Provides data integration services for connecting Streamlit UI
with Spark risk analysis components.
"""

from .risk_service import RiskAnalysisService

__all__ = ["RiskAnalysisService"]