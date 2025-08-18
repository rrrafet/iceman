"""
Risk Schema Utilities
=====================

Utility functions for working with the unified risk result schema,
including advanced validation, migration helpers, and format conversion.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from .schema import RiskResultSchema, AnalysisType, ValidationLevel

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Advanced validation utilities for risk result schemas."""
    
    @staticmethod
    def validate_numerical_consistency(schema: RiskResultSchema, tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Perform advanced numerical consistency checks on risk schema.
        
        Parameters
        ----------
        schema : RiskResultSchema
            Schema to validate
        tolerance : float, default 1e-6
            Numerical tolerance for validation checks
            
        Returns
        -------
        dict
            Detailed validation results
        """
        results = {}
        data = schema.data
        core_metrics = data["core_metrics"]
        
        # Euler identity validation
        if all(core_metrics[k] is not None for k in ["total_risk", "factor_risk_contribution", "specific_risk_contribution"]):
            total_risk = core_metrics["total_risk"]
            factor_risk = core_metrics["factor_risk_contribution"]
            specific_risk = core_metrics["specific_risk_contribution"]
            sum_risk = factor_risk + specific_risk
            
            results["euler_identity"] = {
                "passes": abs(sum_risk - total_risk) < tolerance,
                "difference": abs(sum_risk - total_risk),
                "expected": total_risk,
                "actual": sum_risk,
                "tolerance": tolerance
            }
        
        # Contribution sum validation
        asset_contributions = data["contributions"]["by_asset"]
        if asset_contributions and core_metrics["total_risk"] is not None:
            contrib_sum = sum(asset_contributions.values())
            total_risk = core_metrics["total_risk"]
            
            results["asset_contribution_sum"] = {
                "passes": abs(contrib_sum - total_risk) < tolerance,
                "difference": abs(contrib_sum - total_risk),
                "expected": total_risk,
                "actual": contrib_sum,
                "tolerance": tolerance
            }
        
        # Factor contribution sum validation
        factor_contributions = data["contributions"]["by_factor"]
        if factor_contributions and core_metrics["factor_risk_contribution"] is not None:
            factor_sum = sum(factor_contributions.values())
            expected_factor_risk = core_metrics["factor_risk_contribution"]
            
            results["factor_contribution_sum"] = {
                "passes": abs(factor_sum - expected_factor_risk) < tolerance,
                "difference": abs(factor_sum - expected_factor_risk),
                "expected": expected_factor_risk,
                "actual": factor_sum,
                "tolerance": tolerance
            }
        
        # Percentage validation
        if core_metrics["factor_risk_percentage"] is not None and core_metrics["specific_risk_percentage"] is not None:
            percentage_sum = core_metrics["factor_risk_percentage"] + core_metrics["specific_risk_percentage"]
            
            results["percentage_sum"] = {
                "passes": abs(percentage_sum - 100.0) < tolerance * 100,
                "difference": abs(percentage_sum - 100.0),
                "expected": 100.0,
                "actual": percentage_sum,
                "tolerance": tolerance * 100
            }
        
        # Overall validation
        all_passed = all(result.get("passes", True) for result in results.values())
        results["overall_numerical"] = {
            "passes": all_passed,
            "total_checks": len(results),
            "passed_checks": sum(1 for r in results.values() if r.get("passes", True))
        }
        
        return results
    
    @staticmethod
    def validate_dimension_consistency(schema: RiskResultSchema) -> Dict[str, Any]:
        """
        Validate dimensional consistency across arrays and named dictionaries.
        
        Parameters
        ----------
        schema : RiskResultSchema
            Schema to validate
            
        Returns
        -------
        dict
            Dimension validation results
        """
        results = {}
        data = schema.data
        identifiers = data["identifiers"]
        
        n_assets = len(identifiers["asset_names"])
        n_factors = len(identifiers["factor_names"])
        
        # Asset dimension checks
        asset_contributions = data["contributions"]["by_asset"]
        if asset_contributions:
            actual_assets = len(asset_contributions)
            results["asset_dimensions"] = {
                "passes": actual_assets == n_assets or n_assets == 0,
                "expected": n_assets,
                "actual": actual_assets,
                "message": f"Asset contributions: expected {n_assets}, got {actual_assets}"
            }
        
        # Factor dimension checks
        factor_contributions = data["contributions"]["by_factor"]
        if factor_contributions:
            actual_factors = len(factor_contributions)
            results["factor_dimensions"] = {
                "passes": actual_factors == n_factors or n_factors == 0,
                "expected": n_factors,
                "actual": actual_factors,
                "message": f"Factor contributions: expected {n_factors}, got {actual_factors}"
            }
        
        # Factor exposures dimensions
        factor_exposures = data["exposures"]["factor_exposures"]
        if factor_exposures:
            actual_exposures = len(factor_exposures)
            results["exposure_dimensions"] = {
                "passes": actual_exposures == n_factors or n_factors == 0,
                "expected": n_factors,
                "actual": actual_exposures,
                "message": f"Factor exposures: expected {n_factors}, got {actual_exposures}"
            }
        
        # Factor loadings dimensions
        factor_loadings = data["exposures"]["factor_loadings"]
        if factor_loadings:
            actual_assets_in_loadings = len(factor_loadings)
            results["loadings_asset_dimensions"] = {
                "passes": actual_assets_in_loadings == n_assets or n_assets == 0,
                "expected": n_assets,
                "actual": actual_assets_in_loadings,
                "message": f"Factor loadings assets: expected {n_assets}, got {actual_assets_in_loadings}"
            }
            
            # Check factor dimensions within loadings
            if factor_loadings:
                first_asset_loadings = next(iter(factor_loadings.values()))
                actual_factors_in_loadings = len(first_asset_loadings)
                results["loadings_factor_dimensions"] = {
                    "passes": actual_factors_in_loadings == n_factors or n_factors == 0,
                    "expected": n_factors,
                    "actual": actual_factors_in_loadings,
                    "message": f"Factor loadings factors: expected {n_factors}, got {actual_factors_in_loadings}"
                }
        
        # Overall dimension validation
        all_passed = all(result.get("passes", True) for result in results.values())
        results["overall_dimensions"] = {
            "passes": all_passed,
            "total_checks": len(results),
            "passed_checks": sum(1 for r in results.values() if r.get("passes", True))
        }
        
        return results


class SchemaConverter:
    """Conversion utilities between different risk result formats."""
    
    @staticmethod
    def decomposer_to_schema(decomposer_result: Dict[str, Any]) -> RiskResultSchema:
        """
        Convert RiskDecomposer.to_dict() result to unified schema.
        
        Parameters
        ----------
        decomposer_result : dict
            Result from RiskDecomposer.to_dict()
            
        Returns
        -------
        RiskResultSchema
            Unified schema instance
        """
        return RiskResultSchema.from_decomposer_result(decomposer_result)
    
    @staticmethod
    def strategy_to_schema(strategy_result: Dict[str, Any]) -> RiskResultSchema:
        """
        Convert Strategy.analyze() result to unified schema.
        
        Parameters
        ----------
        strategy_result : dict
            Result from risk analysis strategy
            
        Returns
        -------
        RiskResultSchema
            Unified schema instance
        """
        return RiskResultSchema.from_strategy_result(strategy_result)
    
    @staticmethod
    def portfolio_summary_to_schema(
        summary: Dict[str, Any],
        analysis_type: str = "hierarchical"
    ) -> RiskResultSchema:
        """
        Convert portfolio risk summary to unified schema.
        
        Parameters
        ----------
        summary : dict
            Portfolio risk summary dictionary
        analysis_type : str, default "hierarchical"
            Type of analysis
            
        Returns
        -------
        RiskResultSchema
            Unified schema instance
        """
        schema = RiskResultSchema(
            analysis_type=analysis_type,
            asset_names=summary.get("component_names", []),
            factor_names=summary.get("factor_names", [])
        )
        
        # Set core metrics if available
        if "portfolio_volatility" in summary:
            schema.set_core_metrics(
                summary["portfolio_volatility"],
                summary.get("factor_risk_contribution", 0.0),
                summary.get("specific_risk_contribution", 0.0)
            )
        
        return schema
    
    @staticmethod
    def merge_schemas(schemas: List[RiskResultSchema]) -> RiskResultSchema:
        """
        Merge multiple schemas into a consolidated view.
        
        Parameters
        ----------
        schemas : list of RiskResultSchema
            Schemas to merge
            
        Returns
        -------
        RiskResultSchema
            Merged schema
        """
        if not schemas:
            return RiskResultSchema("portfolio")
        
        # Use first schema as base
        base_schema = schemas[0]
        merged = RiskResultSchema(
            analysis_type="hierarchical",  # Merged is inherently hierarchical
            asset_names=base_schema.asset_names.copy(),
            factor_names=base_schema.factor_names.copy()
        )
        
        # Aggregate core metrics (simple sum for now)
        total_risk = 0.0
        total_factor_risk = 0.0
        total_specific_risk = 0.0
        
        for schema in schemas:
            data = schema.data
            core = data["core_metrics"]
            if core["total_risk"] is not None:
                total_risk += core["total_risk"]
            if core["factor_risk_contribution"] is not None:
                total_factor_risk += core["factor_risk_contribution"]
            if core["specific_risk_contribution"] is not None:
                total_specific_risk += core["specific_risk_contribution"]
        
        if total_risk > 0:
            merged.set_core_metrics(total_risk, total_factor_risk, total_specific_risk)
        
        # Merge contributions (simple aggregation)
        merged_asset_contrib = {}
        merged_factor_contrib = {}
        
        for schema in schemas:
            data = schema.data
            
            # Aggregate asset contributions
            for asset, contrib in data["contributions"]["by_asset"].items():
                merged_asset_contrib[asset] = merged_asset_contrib.get(asset, 0.0) + contrib
            
            # Aggregate factor contributions
            for factor, contrib in data["contributions"]["by_factor"].items():
                merged_factor_contrib[factor] = merged_factor_contrib.get(factor, 0.0) + contrib
        
        if merged_asset_contrib:
            merged.set_asset_contributions(merged_asset_contrib)
        if merged_factor_contrib:
            merged.set_factor_contributions(merged_factor_contrib)
        
        # Add details about merge
        merged.add_detail("merged_from", [str(schema) for schema in schemas])
        merged.add_detail("merge_timestamp", pd.Timestamp.now().isoformat())
        
        return merged


class SchemaMigrator:
    """Migration utilities for transitioning to unified schema."""
    
    @staticmethod
    def create_migration_report(old_format: Dict[str, Any], new_schema: RiskResultSchema) -> Dict[str, Any]:
        """
        Create migration report comparing old and new formats.
        
        Parameters
        ----------
        old_format : dict
            Original format dictionary
        new_schema : RiskResultSchema
            Converted unified schema
            
        Returns
        -------
        dict
            Migration report
        """
        report = {
            "migration_timestamp": pd.Timestamp.now().isoformat(),
            "old_format_keys": list(old_format.keys()),
            "new_schema_sections": list(new_schema.data.keys()),
            "data_preservation": {},
            "improvements": []
        }
        
        # Check data preservation
        if "portfolio_volatility" in old_format:
            old_value = old_format["portfolio_volatility"]
            new_value = new_schema.data["core_metrics"]["total_risk"]
            report["data_preservation"]["portfolio_volatility"] = {
                "preserved": abs(old_value - new_value) < 1e-10 if new_value is not None else False,
                "old_value": old_value,
                "new_value": new_value
            }
        
        # Document improvements
        if new_schema.data["exposures"]["factor_loadings"]:
            report["improvements"].append("Added named factor loadings mapping")
        
        if new_schema.data["contributions"]["by_asset"]:
            report["improvements"].append("Added named asset contributions")
        
        if new_schema.data["validation"]["passes"]:
            report["improvements"].append("Added comprehensive validation")
        
        return report
    
    @staticmethod
    def batch_migrate_results(
        results: List[Dict[str, Any]], 
        source_format: str = "decomposer"
    ) -> List[RiskResultSchema]:
        """
        Migrate multiple results to unified schema format.
        
        Parameters
        ----------
        results : list of dict
            List of results in old format
        source_format : str, default "decomposer"
            Source format type ("decomposer", "strategy", "analyzer")
            
        Returns
        -------
        list of RiskResultSchema
            Migrated schemas
        """
        migrated_schemas = []
        
        for i, result in enumerate(results):
            try:
                if source_format == "decomposer":
                    schema = SchemaConverter.decomposer_to_schema(result)
                elif source_format == "strategy":
                    schema = SchemaConverter.strategy_to_schema(result)
                else:
                    # Generic conversion
                    schema = RiskResultSchema("portfolio")
                    if "portfolio_volatility" in result:
                        schema.set_core_metrics(
                            result["portfolio_volatility"],
                            result.get("factor_risk_contribution", 0.0),
                            result.get("specific_risk_contribution", 0.0)
                        )
                
                migrated_schemas.append(schema)
                
            except Exception as e:
                logger.warning(f"Failed to migrate result {i}: {e}")
                # Create empty schema as fallback
                migrated_schemas.append(RiskResultSchema("portfolio"))
        
        return migrated_schemas


def validate_unified_schema(schema: RiskResultSchema, strict: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Comprehensive validation of unified schema.
    
    Parameters
    ----------
    schema : RiskResultSchema
        Schema to validate
    strict : bool, default False
        Whether to use strict validation mode
        
    Returns
    -------
    tuple
        (passes, validation_results) where passes is bool and validation_results is dict
    """
    all_results = {}
    
    # Basic schema validation
    basic_validation = schema.validate_schema()
    all_results["basic"] = basic_validation
    
    # Numerical consistency validation
    numerical_validation = SchemaValidator.validate_numerical_consistency(schema)
    all_results["numerical"] = numerical_validation
    
    # Dimension consistency validation
    dimension_validation = SchemaValidator.validate_dimension_consistency(schema)
    all_results["dimensions"] = dimension_validation
    
    # Overall validation result
    basic_passes = basic_validation.get("overall", {}).get("passes", False)
    numerical_passes = numerical_validation.get("overall_numerical", {}).get("passes", False)
    dimension_passes = dimension_validation.get("overall_dimensions", {}).get("passes", False)
    
    if strict:
        overall_passes = basic_passes and numerical_passes and dimension_passes
    else:
        # In non-strict mode, only basic validation must pass
        overall_passes = basic_passes
    
    all_results["overall"] = {
        "passes": overall_passes,
        "basic_passes": basic_passes,
        "numerical_passes": numerical_passes,
        "dimension_passes": dimension_passes,
        "validation_mode": "strict" if strict else "moderate"
    }
    
    return overall_passes, all_results


def create_schema_from_arrays(
    total_risk: float,
    factor_contributions: np.ndarray,
    asset_contributions: np.ndarray,
    factor_exposures: Optional[np.ndarray] = None,
    factor_loadings: Optional[np.ndarray] = None,
    asset_names: Optional[List[str]] = None,
    factor_names: Optional[List[str]] = None,
    analysis_type: str = "portfolio"
) -> RiskResultSchema:
    """
    Convenience function to create schema from numpy arrays.
    
    Parameters
    ----------
    total_risk : float
        Total portfolio risk
    factor_contributions : array-like
        Factor risk contributions
    asset_contributions : array-like
        Asset risk contributions
    factor_exposures : array-like, optional
        Portfolio factor exposures
    factor_loadings : array-like, optional
        Asset factor loadings (N×K matrix)
    asset_names : list of str, optional
        Asset names/symbols
    factor_names : list of str, optional
        Factor names
    analysis_type : str, default "portfolio"
        Type of analysis
        
    Returns
    -------
    RiskResultSchema
        Configured schema instance
    """
    schema = RiskResultSchema(
        analysis_type=analysis_type,
        asset_names=asset_names or [],
        factor_names=factor_names or []
    )
    
    # Set core metrics
    factor_risk_total = np.sum(factor_contributions)
    specific_risk_total = total_risk - factor_risk_total  # Approximate
    schema.set_core_metrics(total_risk, factor_risk_total, specific_risk_total)
    
    # Set contributions
    schema.set_factor_contributions(factor_contributions)
    schema.set_asset_contributions(asset_contributions)
    
    # Set exposures if provided
    if factor_exposures is not None:
        schema.set_factor_exposures(factor_exposures)
    
    # Set factor loadings if provided
    if factor_loadings is not None:
        schema.set_factor_loadings(factor_loadings)
    
    return schema