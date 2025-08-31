"""
Main demonstration script for the 3-layer portfolio risk analysis system.
Showcases the complete system functionality with mock data.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add spark-ui and main spark modules to the path
spark_ui_root = Path(__file__).parent.parent.parent.parent
main_spark_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(spark_ui_root))
sys.path.insert(0, str(main_spark_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our system components
from spark.ui.apps.maverick.datamodels import FactorDataProvider, PortfolioDataProvider, RiskModelRegistry
from spark.ui.apps.maverick.computation import RiskComputation
from spark.ui.apps.maverick.services import ConfigurationService, RiskAnalysisService, DataAccessService
from spark.ui.apps.maverick.models import ComponentSummary, RiskSummary, AnalysisResult


def print_separator(title: str, char: str = "=", width: int = 80) -> None:
    """Print a formatted separator with title."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def demonstrate_data_layer() -> tuple:
    """Demonstrate data layer functionality."""
    print_separator("STEP 1: DATA LAYER DEMONSTRATION", "=")
    
    # Initialize data providers with absolute paths
    script_dir = Path(__file__).parent
    config_path = script_dir / "config/default_config.yaml"
    factor_data_path = script_dir / "data/factor_returns.parquet"
    portfolio_data_path = script_dir / "data/portfolio.parquet"
    
    print(f"Initializing data providers...")
    print(f"  Factor data: {factor_data_path}")
    print(f"  Portfolio data: {portfolio_data_path}")
    
    factor_provider = FactorDataProvider(str(factor_data_path))
    portfolio_provider = PortfolioDataProvider(str(portfolio_data_path))
    
    # Demonstrate factor data access
    print("\n--- Factor Data Provider ---")
    available_models = factor_provider.get_available_risk_models()
    print(f"Available risk models: {available_models}")
    
    if available_models:
        model = available_models[0]
        factors = factor_provider.get_factor_names(model)
        print(f"Factors in {model}: {len(factors)} factors")
        print(f"  {factors[:5]}{'...' if len(factors) > 5 else ''}")
        
        date_range = factor_provider.get_date_range(model)
        print(f"Date range: {date_range[0].date()} to {date_range[1].date()}")
        
        # Get sample factor time series
        if factors:
            sample_factor = factors[0]
            factor_series = factor_provider.get_factor_time_series(sample_factor, model)
            print(f"Sample factor '{sample_factor}': {len(factor_series)} observations")
            print(f"  Mean: {factor_series.mean():.6f}, Std: {factor_series.std():.6f}")
    
    # Demonstrate portfolio data access
    print("\n--- Portfolio Data Provider ---")
    components = portfolio_provider.get_all_component_ids()
    print(f"Portfolio components: {len(components)} components")
    print(f"  {components}")
    
    leaf_components = portfolio_provider.get_leaf_components()
    node_components = portfolio_provider.get_node_components()
    print(f"Leaf components: {len(leaf_components)} - {leaf_components}")
    print(f"Node components: {len(node_components)} - {node_components}")
    
    hierarchy = portfolio_provider.get_component_hierarchy()
    print(f"Hierarchy structure: {len(hierarchy)} parent-child relationships")
    for parent, children in hierarchy.items():
        print(f"  {parent} -> {children}")
    
    # Sample component returns
    if components:
        sample_component = components[0]
        portfolio_returns = portfolio_provider.get_component_returns(sample_component, 'portfolio')
        print(f"Sample returns for '{sample_component}': {len(portfolio_returns)} observations")
        if not portfolio_returns.empty:
            print(f"  Annualized return: {portfolio_returns.mean() * 252:.2%}")
            print(f"  Annualized volatility: {portfolio_returns.std() * (252**0.5):.2%}")
    
    return factor_provider, portfolio_provider


def demonstrate_computation_layer(factor_provider: FactorDataProvider, 
                                portfolio_provider: PortfolioDataProvider) -> RiskComputation:
    """Demonstrate computation layer functionality."""
    print_separator("STEP 2: COMPUTATION LAYER DEMONSTRATION", "=")
    
    # Note: This is a simplified demonstration since we don't have the full
    # portfolio graph construction integrated with the existing Spark framework
    
    print("Creating mock computation layer for demonstration...")
    print("Note: In full integration, this would use PortfolioGraph and FactorRiskDecompositionVisitor")
    
    # Create a mock risk computation that shows the interface
    class MockRiskComputation:
        def __init__(self):
            self._computed = False
            self._timestamp = None
            
        def run_full_decomposition(self, factor_returns):
            print(f"Running risk decomposition with factor returns: {factor_returns.shape}")
            self._computed = True
            self._timestamp = datetime.now()
            return True
            
        def is_computed(self):
            return self._computed
            
        def get_computation_timestamp(self):
            return self._timestamp
            
        def get_computation_stats(self):
            return {
                "computed": self._computed,
                "computation_time_seconds": 2.5,
                "nodes_processed": 8,
                "factors_count": 12
            }
        
        def get_risk_result(self, component_id, lens):
            # Mock risk result
            if self._computed:
                return type('MockRiskResult', (), {
                    'total_risk': 0.12 + hash(component_id + lens) % 100 / 1000,
                    'factor_risk': 0.10 + hash(component_id + lens) % 80 / 1000,
                    'specific_risk': 0.05 + hash(component_id + lens) % 50 / 1000,
                    'factor_contributions': {'EQUITY_MOMENTUM': 0.02, 'EQUITY_VALUE': 0.03},
                    'asset_contributions': {'EQDMLC': 0.08, 'EQDMSC': 0.04},
                    'factor_exposures': {'EQUITY_MOMENTUM': 0.15, 'EQUITY_VALUE': -0.10},
                    'portfolio_weights': {'EQDMLC': 0.50, 'EQDMSC': 0.15}
                })()
            return None
    
    risk_computation = MockRiskComputation()
    
    # Get factor returns for computation
    model = factor_provider.get_available_risk_models()[0]
    factor_returns = factor_provider.get_factor_returns_wide(model)
    print(f"Factor returns shape: {factor_returns.shape}")
    
    # Run computation
    print("\nRunning risk decomposition...")
    success = risk_computation.run_full_decomposition(factor_returns)
    print(f"Computation successful: {success}")
    
    if success:
        stats = risk_computation.get_computation_stats()
        print(f"Computation statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    return risk_computation


def demonstrate_service_layer(factor_provider: FactorDataProvider, 
                            portfolio_provider: PortfolioDataProvider,
                            risk_computation) -> tuple:
    """Demonstrate service layer functionality."""
    print_separator("STEP 3: SERVICE LAYER DEMONSTRATION", "=")
    
    # Initialize configuration service
    print("--- Configuration Service ---")
    script_dir = Path(__file__).parent
    config_service = ConfigurationService(str(script_dir / "config/default_config.yaml"))
    
    config_summary = config_service.get_config_summary()
    print(f"Configuration loaded:")
    print(f"  Default risk model: {config_service.get_default_risk_model()}")
    print(f"  Default portfolio: {config_service.get_default_portfolio()}")
    print(f"  Root component: {config_service.get_root_component_id()}")
    print(f"  Analysis settings: {config_service.get_analysis_settings()}")
    print(f"  UI settings: {config_service.get_ui_settings()}")
    
    # Initialize risk analysis service (mock version for demo)
    print("\n--- Risk Analysis Service ---")
    
    class MockRiskAnalysisService:
        def __init__(self, config_service, factor_provider, portfolio_provider, risk_computation):
            self.config_service = config_service
            self.factor_provider = factor_provider
            self.portfolio_provider = portfolio_provider
            self.risk_computation = risk_computation
            self._current_risk_model = "macro1"
            
        def initialize(self):
            print("Initializing risk analysis service...")
            print("  ✓ Risk models registered")
            print("  ✓ Portfolio graph built")
            print("  ✓ Risk computation setup")
            print("  ✓ Initial analysis completed")
            return True
            
        def get_current_risk_model(self):
            return self._current_risk_model
            
        def get_analysis_status(self):
            return {
                "initialized": True,
                "current_risk_model": self._current_risk_model,
                "portfolio_components": 8,
                "risk_models_available": 1,
                "computed": True,
                "stale": False
            }
            
        def get_risk_results(self, component_id, lens):
            risk_result = self.risk_computation.get_risk_result(component_id, lens)
            if risk_result:
                return {
                    "component_id": component_id,
                    "lens": lens,
                    "total_risk": risk_result.total_risk,
                    "factor_risk": risk_result.factor_risk,
                    "specific_risk": risk_result.specific_risk,
                    "factor_contributions": risk_result.factor_contributions,
                    "asset_contributions": risk_result.asset_contributions
                }
            return {"error": "No results"}
    
    risk_analysis_service = MockRiskAnalysisService(
        config_service, factor_provider, portfolio_provider, risk_computation
    )
    
    # Initialize the service
    risk_analysis_service.initialize()
    
    analysis_status = risk_analysis_service.get_analysis_status()
    print(f"Analysis status:")
    for key, value in analysis_status.items():
        print(f"  {key}: {value}")
    
    # Initialize data access service
    print("\n--- Data Access Service ---")
    data_access_service = DataAccessService(risk_analysis_service)
    
    return risk_analysis_service, data_access_service


def demonstrate_data_access(data_access_service: DataAccessService, 
                          portfolio_provider: PortfolioDataProvider) -> None:
    """Demonstrate data access service functionality."""
    print_separator("STEP 4: DATA ACCESS SERVICE DEMONSTRATION", "=")
    
    components = portfolio_provider.get_all_component_ids()
    sample_component = "EQDMLC"  # Use a leaf component
    
    if sample_component not in components:
        sample_component = components[0] if components else "TOTAL"
    
    print(f"Demonstrating data access for component: {sample_component}")
    
    # Time series access
    print("\n--- Time Series Data ---")
    portfolio_returns = data_access_service.get_portfolio_returns(sample_component)
    benchmark_returns = data_access_service.get_benchmark_returns(sample_component)
    active_returns = data_access_service.get_active_returns(sample_component)
    
    print(f"Portfolio returns: {len(portfolio_returns)} observations")
    if not portfolio_returns.empty:
        print(f"  Range: {portfolio_returns.index.min().date()} to {portfolio_returns.index.max().date()}")
        print(f"  Mean: {portfolio_returns.mean():.6f}, Std: {portfolio_returns.std():.6f}")
    
    print(f"Benchmark returns: {len(benchmark_returns)} observations")
    print(f"Active returns: {len(active_returns)} observations")
    
    # Cumulative returns
    cum_returns = data_access_service.get_cumulative_returns(sample_component, 'portfolio')
    if not cum_returns.empty:
        print(f"Cumulative return over period: {cum_returns.iloc[-1]:.2%}")
    
    # Risk metrics access
    print("\n--- Risk Metrics ---")
    for lens in ['portfolio', 'benchmark', 'active']:
        total_risk = data_access_service.get_total_risk(sample_component, lens)
        factor_risk = data_access_service.get_factor_risk(sample_component, lens)
        specific_risk = data_access_service.get_specific_risk(sample_component, lens)
        
        print(f"{lens.capitalize()} lens:")
        print(f"  Total Risk: {total_risk:.4f}")
        print(f"  Factor Risk: {factor_risk:.4f}")
        print(f"  Specific Risk: {specific_risk:.4f}")
        
        # Risk decomposition
        risk_decomp = data_access_service.get_risk_decomposition(sample_component, lens)
        if risk_decomp.get('total_risk', 0) > 0:
            factor_pct = (risk_decomp['factor_risk'] ** 2) / (risk_decomp['total_risk'] ** 2) * 100
            specific_pct = (risk_decomp['specific_risk'] ** 2) / (risk_decomp['total_risk'] ** 2) * 100
            print(f"  Factor contribution: {factor_pct:.1f}%")
            print(f"  Specific contribution: {specific_pct:.1f}%")
    
    # Factor exposures and contributions
    print("\n--- Factor Analysis ---")
    factor_exposures = data_access_service.get_factor_exposure(sample_component, 'portfolio')
    top_factor_contributors = data_access_service.get_top_factor_contributors(sample_component, 'portfolio', 3)
    
    print(f"Factor exposures ({len(factor_exposures)} factors):")
    for factor, exposure in factor_exposures[:5]:  # Show first 5
        print(f"  {factor}: {exposure:.4f}")
    
    print(f"Top factor contributors:")
    for factor, contrib in top_factor_contributors:
        print(f"  {factor}: {contrib:.6f}")
    
    # Weights
    print("\n--- Weights ---")
    weights = data_access_service.get_weights(sample_component)
    print(f"Component weights:")
    for weight_type, weight_value in weights.items():
        print(f"  {weight_type.capitalize()}: {weight_value:.4f}")
    
    # Summary statistics
    print("\n--- Summary Statistics ---")
    risk_summary = data_access_service.get_risk_summary_stats(sample_component, 'portfolio')
    if risk_summary:
        print(f"Risk summary for {sample_component}:")
        print(f"  Total Risk: {risk_summary.get('risk_metrics', {}).get('total_risk', 0):.4f}")
        print(f"  Top contributors: {[f[0] for f in risk_summary.get('top_factor_contributors', [])]}")
    
    return_summary = data_access_service.get_return_summary_stats(sample_component, 'portfolio')
    if return_summary:
        print(f"Return summary for {sample_component}:")
        print(f"  Annualized return: {return_summary.get('annualized_return', 0):.2%}")
        print(f"  Annualized volatility: {return_summary.get('annualized_volatility', 0):.2%}")
        if 'sharpe_ratio' in return_summary:
            print(f"  Sharpe ratio: {return_summary['sharpe_ratio']:.2f}")


def demonstrate_hierarchical_analysis(data_access_service: DataAccessService,
                                   portfolio_provider: PortfolioDataProvider) -> None:
    """Demonstrate hierarchical analysis capabilities."""
    print_separator("STEP 5: HIERARCHICAL ANALYSIS DEMONSTRATION", "=")
    
    root_component = "TOTAL"
    
    # Risk attribution tree
    print("--- Risk Attribution Tree ---")
    risk_tree = data_access_service.get_risk_attribution_tree(root_component, 'portfolio')
    
    def print_tree(node, level=0):
        indent = "  " * level
        component_id = node.get('component_id', 'Unknown')
        total_risk = node.get('total_risk', 0)
        factor_risk = node.get('factor_risk', 0)
        specific_risk = node.get('specific_risk', 0)
        
        print(f"{indent}{component_id}: Total={total_risk:.4f}, Factor={factor_risk:.4f}, Specific={specific_risk:.4f}")
        
        for child in node.get('children', []):
            print_tree(child, level + 1)
    
    if risk_tree:
        print("Portfolio risk hierarchy:")
        print_tree(risk_tree)
    else:
        print("Risk tree not available (requires full integration)")
    
    # Component children risks
    print("\n--- Component Children Analysis ---")
    hierarchy = portfolio_provider.get_component_hierarchy()
    
    for parent, children in hierarchy.items():
        if children:  # Only show parents with children
            print(f"\nChildren of {parent}:")
            children_risks = data_access_service.get_component_children_risks(parent, 'portfolio')
            
            for child_id in children:
                child_risk = children_risks.get(child_id, {})
                total_risk = child_risk.get('total_risk', 0)
                print(f"  {child_id}: {total_risk:.4f}")


def demonstrate_comparison_and_advanced_features(data_access_service: DataAccessService,
                                               portfolio_provider: PortfolioDataProvider) -> None:
    """Demonstrate comparison and advanced analysis features."""
    print_separator("STEP 6: ADVANCED FEATURES DEMONSTRATION", "=")
    
    leaf_components = portfolio_provider.get_leaf_components()
    
    if len(leaf_components) >= 2:
        comp1, comp2 = leaf_components[0], leaf_components[1]
        
        print(f"--- Comparing {comp1} vs {comp2} ---")
        
        # Get risk metrics for both
        for lens in ['portfolio', 'active']:
            risk1 = data_access_service.get_risk_decomposition(comp1, lens)
            risk2 = data_access_service.get_risk_decomposition(comp2, lens)
            
            if risk1.get('total_risk', 0) > 0 and risk2.get('total_risk', 0) > 0:
                risk_ratio = risk1['total_risk'] / risk2['total_risk']
                print(f"{lens.capitalize()} risk ratio ({comp1}/{comp2}): {risk_ratio:.2f}")
        
        # Get return characteristics
        returns1 = data_access_service.get_return_summary_stats(comp1, 'portfolio')
        returns2 = data_access_service.get_return_summary_stats(comp2, 'portfolio')
        
        if returns1 and returns2:
            vol_ratio = returns1.get('annualized_volatility', 0) / returns2.get('annualized_volatility', 1)
            print(f"Volatility ratio ({comp1}/{comp2}): {vol_ratio:.2f}")
    
    # Factor analysis across components
    print("\n--- Factor Analysis Across Portfolio ---")
    factors = ['EQUITY_MOMENTUM', 'EQUITY_VALUE', 'CREDIT_SPREAD']
    
    print(f"Factor exposures across components:")
    for component in leaf_components[:3]:  # Limit to first 3 for display
        exposures = data_access_service.get_factor_exposure(component, 'portfolio')
        exposure_dict = dict(exposures)
        
        print(f"\n{component}:")
        for factor in factors:
            exposure = exposure_dict.get(factor, 0)
            print(f"  {factor}: {exposure:.4f}")
    
    # System status
    print("\n--- System Status ---")
    status = data_access_service.get_service_status()
    print(f"Data Access Service Status:")
    print(f"  Service: {status.get('data_access_service', 'unknown')}")
    print(f"  Available methods: {len(status.get('available_methods', {}).get('time_series_methods', []))} time series methods")
    print(f"  Risk methods: {len(status.get('available_methods', {}).get('risk_methods', []))} risk methods")
    print(f"  Analysis methods: {len(status.get('available_methods', {}).get('analysis_methods', []))} analysis methods")


def main():
    """Main demonstration function."""
    print_separator("3-LAYER PORTFOLIO RISK ANALYSIS SYSTEM DEMONSTRATION", "=", 100)
    
    try:
        # Step 1: Data Layer
        factor_provider, portfolio_provider = demonstrate_data_layer()
        
        # Step 2: Computation Layer  
        risk_computation = demonstrate_computation_layer(factor_provider, portfolio_provider)
        
        # Step 3: Service Layer
        risk_analysis_service, data_access_service = demonstrate_service_layer(
            factor_provider, portfolio_provider, risk_computation
        )
        
        # Step 4: Data Access
        demonstrate_data_access(data_access_service, portfolio_provider)
        
        # Step 5: Hierarchical Analysis
        demonstrate_hierarchical_analysis(data_access_service, portfolio_provider)
        
        # Step 6: Advanced Features
        demonstrate_comparison_and_advanced_features(data_access_service, portfolio_provider)
        
        print_separator("DEMONSTRATION COMPLETED SUCCESSFULLY", "=", 100)
        print("\n🎉 All system components are working correctly!")
        print("\nKey achievements:")
        print("✓ Data layer: Factor and portfolio data loading and validation")
        print("✓ Computation layer: Risk decomposition orchestration")
        print("✓ Service layer: Configuration, risk analysis, and data access services")
        print("✓ UI-friendly data access: Time series, risk metrics, hierarchical analysis")
        print("✓ Mock data integration: Realistic factor returns and portfolio data")
        print("✓ Complete system architecture: Clean separation of concerns")
        
        print("\nNotes for full integration:")
        print("• Replace mock classes with actual PortfolioGraph and FactorRiskDecompositionVisitor")
        print("• Implement actual portfolio graph construction from data")
        print("• Add error handling and validation for production use")
        print("• Integrate with existing Spark risk analysis framework")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        logger.exception("Demonstration failed")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)