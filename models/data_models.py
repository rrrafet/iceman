"""
Data models for portfolio risk analysis system.
Dataclasses for structured return types and UI data formatting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd


@dataclass
class ComponentSummary:
    """Summary information for a portfolio component."""
    component_id: str
    name: Optional[str] = None
    component_type: str = "unknown"  # 'leaf', 'node', 'root'
    is_leaf: bool = False
    children_count: int = 0
    children_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    level: int = 0
    path: str = ""


@dataclass 
class FactorContribution:
    """Factor contribution data."""
    factor_name: str
    contribution: float
    exposure: float
    percentage_of_total: Optional[float] = None
    rank: Optional[int] = None


@dataclass
class RiskSummary:
    """Risk analysis summary for a component and lens."""
    component_id: str
    lens: str  # 'portfolio', 'benchmark', 'active'
    total_risk: float
    factor_risk: float
    specific_risk: float
    
    # Decomposition percentages
    factor_risk_pct: Optional[float] = None
    specific_risk_pct: Optional[float] = None
    
    # Top contributors
    top_factor_contributions: List[FactorContribution] = field(default_factory=list)
    top_asset_contributions: List[Tuple[str, float]] = field(default_factory=list)
    
    # Weights
    portfolio_weight: Optional[float] = None
    benchmark_weight: Optional[float] = None
    active_weight: Optional[float] = None
    
    # Validation
    euler_identity_check: Optional[float] = None  # Relative error
    is_valid: bool = True
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.total_risk > 0:
            self.factor_risk_pct = (self.factor_risk ** 2) / (self.total_risk ** 2) * 100
            self.specific_risk_pct = (self.specific_risk ** 2) / (self.total_risk ** 2) * 100
        
        # Calculate Euler identity check
        if self.total_risk > 0:
            expected_total_risk_sq = self.factor_risk ** 2 + self.specific_risk ** 2
            actual_total_risk_sq = self.total_risk ** 2
            self.euler_identity_check = abs(actual_total_risk_sq - expected_total_risk_sq) / actual_total_risk_sq
            self.is_valid = self.euler_identity_check < 0.01  # 1% tolerance


@dataclass
class TimeSeriesData:
    """Time series data container."""
    component_id: str
    data_type: str  # 'returns', 'cumulative_returns', 'volatility'
    return_type: str  # 'portfolio', 'benchmark', 'active'
    series: pd.Series
    
    # Summary statistics
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    count: Optional[int] = None
    
    # Annualized metrics
    annualized_return: Optional[float] = None
    annualized_volatility: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    
    # Date range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def __post_init__(self):
        """Calculate summary statistics after initialization."""
        if not self.series.empty:
            self.mean = float(self.series.mean())
            self.std = float(self.series.std())
            self.min_value = float(self.series.min())
            self.max_value = float(self.series.max())
            self.count = len(self.series)
            
            # Date range
            if hasattr(self.series.index, 'min'):
                self.start_date = self.series.index.min().to_pydatetime()
                self.end_date = self.series.index.max().to_pydatetime()
            
            # Annualized metrics (assuming daily data)
            if self.data_type == 'returns' and self.std:
                self.annualized_return = self.mean * 252
                self.annualized_volatility = self.std * (252 ** 0.5)
                
                if self.return_type != 'active' and self.annualized_volatility > 0:
                    self.sharpe_ratio = self.annualized_return / self.annualized_volatility


@dataclass
class AnalysisResult:
    """Complete analysis result for a component."""
    component_summary: ComponentSummary
    risk_summaries: Dict[str, RiskSummary] = field(default_factory=dict)  # lens -> RiskSummary
    time_series_data: Dict[str, TimeSeriesData] = field(default_factory=dict)  # key -> TimeSeriesData
    
    # Hierarchical data (if applicable)
    children_analysis: Dict[str, 'AnalysisResult'] = field(default_factory=dict)
    parent_analysis: Optional['AnalysisResult'] = None
    
    # Metadata
    analysis_timestamp: Optional[datetime] = None
    risk_model_used: Optional[str] = None
    computation_time_seconds: Optional[float] = None
    
    def get_risk_summary(self, lens: str) -> Optional[RiskSummary]:
        """Get risk summary for a specific lens."""
        return self.risk_summaries.get(lens)
    
    def get_time_series(self, key: str) -> Optional[TimeSeriesData]:
        """Get time series data by key."""
        return self.time_series_data.get(key)
    
    def add_risk_summary(self, lens: str, risk_summary: RiskSummary) -> None:
        """Add risk summary for a lens."""
        self.risk_summaries[lens] = risk_summary
    
    def add_time_series(self, key: str, time_series_data: TimeSeriesData) -> None:
        """Add time series data."""
        self.time_series_data[key] = time_series_data
    
    def is_valid(self) -> bool:
        """Check if all risk summaries are valid."""
        return all(summary.is_valid for summary in self.risk_summaries.values())
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get a summary dictionary for UI display."""
        summary = {
            "component_id": self.component_summary.component_id,
            "component_type": self.component_summary.component_type,
            "is_leaf": self.component_summary.is_leaf,
            "children_count": self.component_summary.children_count,
            "risk_model": self.risk_model_used,
            "analysis_timestamp": self.analysis_timestamp.isoformat() if self.analysis_timestamp else None,
            "computation_time": self.computation_time_seconds,
            "is_valid": self.is_valid()
        }
        
        # Add risk summaries
        summary["risk_summaries"] = {}
        for lens, risk_summary in self.risk_summaries.items():
            summary["risk_summaries"][lens] = {
                "total_risk": risk_summary.total_risk,
                "factor_risk": risk_summary.factor_risk,
                "specific_risk": risk_summary.specific_risk,
                "factor_risk_pct": risk_summary.factor_risk_pct,
                "specific_risk_pct": risk_summary.specific_risk_pct,
                "is_valid": risk_summary.is_valid,
                "euler_check": risk_summary.euler_identity_check
            }
        
        # Add time series summaries
        summary["time_series"] = {}
        for key, ts_data in self.time_series_data.items():
            summary["time_series"][key] = {
                "data_type": ts_data.data_type,
                "return_type": ts_data.return_type,
                "count": ts_data.count,
                "annualized_return": ts_data.annualized_return,
                "annualized_volatility": ts_data.annualized_volatility,
                "sharpe_ratio": ts_data.sharpe_ratio,
                "date_range": {
                    "start": ts_data.start_date.isoformat() if ts_data.start_date else None,
                    "end": ts_data.end_date.isoformat() if ts_data.end_date else None
                }
            }
        
        return summary


@dataclass
class ComparisonResult:
    """Result of comparing two components."""
    component1: ComponentSummary
    component2: ComponentSummary
    lens: str
    
    # Risk comparison
    risk_diff: Dict[str, float] = field(default_factory=dict)  # 'total_risk', 'factor_risk', 'specific_risk'
    risk_ratio: Dict[str, float] = field(default_factory=dict)  # component1 / component2
    
    # Return comparison (if time series available)
    return_diff: Optional[Dict[str, float]] = None
    correlation: Optional[float] = None
    tracking_error: Optional[float] = None
    
    # Factor exposure differences
    factor_exposure_diff: Dict[str, float] = field(default_factory=dict)
    
    # Weight comparison
    weight_diff: Optional[Dict[str, float]] = None  # 'portfolio', 'benchmark', 'active'
    
    def get_relative_risk(self, risk_type: str) -> Optional[float]:
        """Get relative risk (component1 / component2) for a risk type."""
        return self.risk_ratio.get(risk_type)
    
    def get_risk_difference(self, risk_type: str) -> Optional[float]:
        """Get absolute risk difference (component1 - component2) for a risk type."""
        return self.risk_diff.get(risk_type)


@dataclass
class PortfolioAnalysis:
    """Complete portfolio analysis result."""
    portfolio_name: str
    root_component_id: str
    risk_model_code: str
    analysis_timestamp: datetime
    
    # Component analyses
    component_analyses: Dict[str, AnalysisResult] = field(default_factory=dict)
    
    # Portfolio-level metrics
    total_components: int = 0
    leaf_components: int = 0
    node_components: int = 0
    hierarchy_depth: int = 0
    
    # Summary statistics across all components
    portfolio_risk_range: Optional[Tuple[float, float]] = None  # (min, max) total risk
    factor_risk_contribution_avg: Optional[float] = None
    specific_risk_contribution_avg: Optional[float] = None
    
    # Top contributors at portfolio level
    top_risk_contributors: List[Tuple[str, float]] = field(default_factory=list)
    top_factor_exposures: List[Tuple[str, float]] = field(default_factory=list)
    
    # Validation summary
    valid_components: int = 0
    invalid_components: int = 0
    validation_errors: List[str] = field(default_factory=list)
    
    def add_component_analysis(self, component_id: str, analysis: AnalysisResult) -> None:
        """Add analysis result for a component."""
        self.component_analyses[component_id] = analysis
        analysis.analysis_timestamp = self.analysis_timestamp
        analysis.risk_model_used = self.risk_model_code
    
    def get_component_analysis(self, component_id: str) -> Optional[AnalysisResult]:
        """Get analysis result for a component."""
        return self.component_analyses.get(component_id)
    
    def calculate_summary_statistics(self) -> None:
        """Calculate portfolio-level summary statistics."""
        if not self.component_analyses:
            return
        
        self.total_components = len(self.component_analyses)
        
        # Count leaf/node components
        for analysis in self.component_analyses.values():
            if analysis.component_summary.is_leaf:
                self.leaf_components += 1
            else:
                self.node_components += 1
        
        # Risk range and averages (using portfolio lens)
        portfolio_risks = []
        factor_contribs = []
        specific_contribs = []
        valid_count = 0
        
        for analysis in self.component_analyses.values():
            risk_summary = analysis.get_risk_summary('portfolio')
            if risk_summary:
                portfolio_risks.append(risk_summary.total_risk)
                if risk_summary.factor_risk_pct is not None:
                    factor_contribs.append(risk_summary.factor_risk_pct)
                if risk_summary.specific_risk_pct is not None:
                    specific_contribs.append(risk_summary.specific_risk_pct)
                
                if risk_summary.is_valid:
                    valid_count += 1
        
        if portfolio_risks:
            self.portfolio_risk_range = (min(portfolio_risks), max(portfolio_risks))
        
        if factor_contribs:
            self.factor_risk_contribution_avg = sum(factor_contribs) / len(factor_contribs)
        
        if specific_contribs:
            self.specific_risk_contribution_avg = sum(specific_contribs) / len(specific_contribs)
        
        self.valid_components = valid_count
        self.invalid_components = self.total_components - valid_count
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        self.calculate_summary_statistics()
        
        return {
            "portfolio_name": self.portfolio_name,
            "root_component_id": self.root_component_id,
            "risk_model_code": self.risk_model_code,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "component_counts": {
                "total": self.total_components,
                "leaf": self.leaf_components,
                "node": self.node_components
            },
            "hierarchy_depth": self.hierarchy_depth,
            "risk_statistics": {
                "portfolio_risk_range": self.portfolio_risk_range,
                "avg_factor_contribution_pct": self.factor_risk_contribution_avg,
                "avg_specific_contribution_pct": self.specific_risk_contribution_avg
            },
            "validation": {
                "valid_components": self.valid_components,
                "invalid_components": self.invalid_components,
                "validation_rate": self.valid_components / self.total_components if self.total_components > 0 else 0,
                "errors": self.validation_errors
            },
            "top_contributors": {
                "risk_contributors": self.top_risk_contributors,
                "factor_exposures": self.top_factor_exposures
            }
        }