"""
Risk Result Schema - Standardized data container for risk analysis results.

This module provides the RiskResultSchema class that serves as a comprehensive
data container for risk decomposition results from portfolio hierarchy analysis.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import json
from enum import Enum


class AnalysisType(Enum):
    """Enumeration of different analysis types for risk decomposition."""
    PORTFOLIO = "portfolio"
    BENCHMARK = "benchmark"
    ACTIVE = "active"
    HIERARCHICAL = "hierarchical"


class ValidationLevel(Enum):
    """Enumeration of validation levels for schema validation."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


class RiskResultSchema:
    """
    Comprehensive schema for storing and accessing risk analysis results.
    
    This class provides a standardized interface for risk data extracted from
    portfolio visitors, supporting hierarchical portfolio structures with
    multiple analysis lenses (portfolio, benchmark, active).
    
    Key Features:
    - Hierarchical data storage for complex portfolio structures
    - Multi-lens analysis (portfolio/benchmark/active) 
    - Comprehensive risk metrics (36+ properties per component per lens)
    - Time series data integration
    - Factor and asset attribution data
    - Validation and metadata management
    - Export capabilities
    
    Data Structure:
    ```
    hierarchical_data = {
        "component_id": {
            "portfolio": {comprehensive_risk_data},
            "benchmark": {comprehensive_risk_data},
            "active": {comprehensive_risk_data + active_specific}
        }
    }
    ```
    """
    
    def __init__(
        self,
        analysis_type: Union[str, AnalysisType] = "hierarchical",
        extraction_method: str = "direct_mapping",
        metadata: Optional[Dict[str, Any]] = None,
        asset_names: Optional[List[str]] = None,
        factor_names: Optional[List[str]] = None,
        data_frequency: Optional[str] = None,
        annualized: bool = False
    ):
        """
        Initialize risk result schema.
        
        Parameters
        ----------
        analysis_type : Union[str, AnalysisType], default "hierarchical"
            Type of risk analysis performed
        extraction_method : str, default "direct_mapping"  
            Method used to extract data from source
        metadata : Dict[str, Any], optional
            Additional metadata about the analysis
        asset_names : List[str], optional
            List of asset names in the portfolio
        factor_names : List[str], optional
            List of risk factor names
        data_frequency : str, optional
            Frequency of the data (e.g., 'D' for daily, 'B' for business days)
        annualized : bool, default False
            Whether risk metrics should be annualized
        """
        # Handle both string and enum inputs
        if isinstance(analysis_type, AnalysisType):
            self.analysis_type = analysis_type
        else:
            # Convert string to enum if possible, otherwise keep as string
            try:
                self.analysis_type = AnalysisType(analysis_type)
            except ValueError:
                self.analysis_type = analysis_type
        
        self.extraction_method = extraction_method
        self.metadata = metadata or {}
        self.creation_timestamp = datetime.now()
        
        # Additional parameters
        self.asset_names = asset_names or []
        self.factor_names = factor_names or []
        self.data_frequency = data_frequency
        self.annualized = annualized
        
        # Core data storage
        self._hierarchical_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._time_series_data: Dict[str, pd.DataFrame] = {}
        self._factor_analysis: Dict[str, Any] = {}
        self._global_metadata: Dict[str, Any] = {}
        
        # Legacy data structure for backward compatibility with strategies
        self._data: Dict[str, Any] = {
            'weights': {
                'portfolio_weights': {},
                'benchmark_weights': {},
                'active_weights': {}
            },
            'portfolio': {
                'core_metrics': {},
                'matrices': {},
                'arrays': {}
            },
            'benchmark': {
                'core_metrics': {},
                'matrices': {},
                'arrays': {}
            },
            'active': {
                'core_metrics': {},
                'matrices': {},
                'arrays': {}
            }
        }
        
        # Validation state
        self._validation_results: Optional[Dict[str, Any]] = None
        
    def set_hierarchical_risk_data(self, data: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Set the hierarchical risk data structure.
        
        Parameters
        ----------
        data : Dict[str, Dict[str, Dict[str, Any]]]
            Hierarchical data in format: {component_id: {lens: {risk_data}}}
        """
        self._hierarchical_data = data.copy() if data else {}
        
    def get_hierarchical_risk_data(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get the complete hierarchical risk data structure.
        
        Returns
        -------
        Dict[str, Dict[str, Dict[str, Any]]]
            Complete hierarchical data structure
        """
        return self._hierarchical_data.copy()
    
    def get_component_data(self, component_id: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Get all lens data for a specific component.
        
        Parameters
        ----------
        component_id : str
            Component identifier
            
        Returns
        -------
        Dict[str, Dict[str, Any]] or None
            Lens data for the component, None if not found
        """
        return self._hierarchical_data.get(component_id)
    
    def get_component_lens_data(self, component_id: str, lens: str) -> Optional[Dict[str, Any]]:
        """
        Get specific lens data for a component.
        
        Parameters
        ----------
        component_id : str
            Component identifier
        lens : str
            Analysis lens ('portfolio', 'benchmark', or 'active')
            
        Returns
        -------
        Dict[str, Any] or None
            Lens-specific risk data, None if not found
        """
        component_data = self._hierarchical_data.get(component_id)
        if component_data:
            return component_data.get(lens)
        return None
    
    def get_all_components(self) -> List[str]:
        """
        Get list of all component IDs in the schema.
        
        Returns
        -------
        List[str]
            List of component identifiers
        """
        return list(self._hierarchical_data.keys())
    
    def get_available_lenses(self, component_id: str) -> List[str]:
        """
        Get available analysis lenses for a component.
        
        Parameters
        ----------
        component_id : str
            Component identifier
            
        Returns
        -------
        List[str]
            Available lenses for the component
        """
        component_data = self._hierarchical_data.get(component_id)
        if component_data:
            return list(component_data.keys())
        return []
    
    def set_time_series_data(self, time_series: Dict[str, pd.DataFrame]) -> None:
        """
        Set time series data for the analysis.
        
        Parameters
        ----------
        time_series : Dict[str, pd.DataFrame]
            Time series data indexed by data type
        """
        self._time_series_data = time_series.copy() if time_series else {}
    
    def get_time_series_data(self, series_type: Optional[str] = None) -> Union[Dict[str, pd.DataFrame], pd.DataFrame, None]:
        """
        Get time series data.
        
        Parameters
        ----------
        series_type : str, optional
            Specific time series type to retrieve
            
        Returns
        -------
        Dict[str, pd.DataFrame] or pd.DataFrame or None
            Time series data
        """
        if series_type:
            return self._time_series_data.get(series_type)
        return self._time_series_data.copy()
    
    def set_factor_analysis(self, factor_analysis: Dict[str, Any]) -> None:
        """
        Set factor analysis results.
        
        Parameters
        ----------
        factor_analysis : Dict[str, Any]
            Factor analysis results and metadata
        """
        self._factor_analysis = factor_analysis.copy() if factor_analysis else {}
    
    def get_factor_analysis(self) -> Dict[str, Any]:
        """
        Get factor analysis results.
        
        Returns
        -------
        Dict[str, Any]
            Factor analysis results
        """
        return self._factor_analysis.copy()
    
    def set_global_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set global metadata for the analysis.
        
        Parameters
        ----------
        metadata : Dict[str, Any]
            Global analysis metadata
        """
        self._global_metadata = metadata.copy() if metadata else {}
    
    def get_global_metadata(self) -> Dict[str, Any]:
        """
        Get global analysis metadata.
        
        Returns
        -------
        Dict[str, Any]
            Global metadata
        """
        return self._global_metadata.copy()
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the schema data for consistency and completeness.
        
        Returns
        -------
        Dict[str, Any]
            Validation results with status and any issues found
        """
        issues = []
        warnings = []
        
        # Check hierarchical data structure
        if not self._hierarchical_data:
            issues.append("No hierarchical data found")
        else:
            # Validate each component
            for component_id, component_data in self._hierarchical_data.items():
                if not isinstance(component_data, dict):
                    issues.append(f"Component {component_id}: data is not a dictionary")
                    continue
                    
                # Check for at least one lens
                if not component_data:
                    warnings.append(f"Component {component_id}: no lens data found")
                    continue
                
                # Validate lens data
                valid_lenses = {'portfolio', 'benchmark', 'active'}
                for lens, lens_data in component_data.items():
                    if lens not in valid_lenses:
                        warnings.append(f"Component {component_id}: unexpected lens '{lens}'")
                    
                    if not isinstance(lens_data, dict):
                        issues.append(f"Component {component_id}, lens {lens}: data is not a dictionary")
                        continue
                    
                    # Check for core metrics
                    expected_metrics = ['total_risk', 'factor_contributions', 'asset_contributions']
                    missing_metrics = [m for m in expected_metrics if m not in lens_data]
                    if missing_metrics:
                        warnings.append(f"Component {component_id}, lens {lens}: missing metrics {missing_metrics}")
        
        # Validate time series data consistency
        if self._time_series_data:
            dates = None
            for series_name, df in self._time_series_data.items():
                if not isinstance(df, pd.DataFrame):
                    issues.append(f"Time series {series_name}: not a DataFrame")
                    continue
                
                if dates is None:
                    dates = df.index
                elif not dates.equals(df.index):
                    warnings.append(f"Time series {series_name}: index mismatch with other series")
        
        # Store validation results
        self._validation_results = {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'components_count': len(self._hierarchical_data),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        return self._validation_results
    
    def get_validation_results(self) -> Optional[Dict[str, Any]]:
        """
        Get cached validation results.
        
        Returns
        -------
        Dict[str, Any] or None
            Validation results if validation has been run
        """
        return self._validation_results
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export schema to dictionary format.
        
        Returns
        -------
        Dict[str, Any]
            Complete schema data as dictionary
        """
        return {
            'analysis_type': self.analysis_type,
            'extraction_method': self.extraction_method,
            'metadata': self.metadata,
            'creation_timestamp': self.creation_timestamp.isoformat(),
            'hierarchical_data': self._hierarchical_data,
            'time_series_data': {
                name: df.to_dict('records') for name, df in self._time_series_data.items()
            },
            'factor_analysis': self._factor_analysis,
            'global_metadata': self._global_metadata,
            'validation_results': self._validation_results
        }
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export schema to JSON format.
        
        Parameters
        ----------
        filepath : str, optional
            Path to save JSON file. If None, returns JSON string.
            
        Returns
        -------
        str
            JSON representation of schema
        """
        # Convert numpy arrays to lists for JSON serialization
        data_dict = self.to_dict()
        json_data = self._convert_numpy_for_json(data_dict)
        
        json_str = json.dumps(json_data, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def _convert_numpy_for_json(self, obj):
        """Recursively convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the schema contents.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics and information
        """
        summary = {
            'analysis_type': self.analysis_type,
            'extraction_method': self.extraction_method,
            'creation_timestamp': self.creation_timestamp.isoformat(),
            'components_count': len(self._hierarchical_data),
            'time_series_count': len(self._time_series_data),
            'has_factor_analysis': bool(self._factor_analysis),
            'has_global_metadata': bool(self._global_metadata)
        }
        
        # Component lens breakdown
        lens_counts = {}
        for component_data in self._hierarchical_data.values():
            for lens in component_data.keys():
                lens_counts[lens] = lens_counts.get(lens, 0) + 1
        
        summary['lens_counts'] = lens_counts
        
        # Add validation summary if available
        if self._validation_results:
            summary['validation_status'] = {
                'valid': self._validation_results['valid'],
                'issues_count': len(self._validation_results['issues']),
                'warnings_count': len(self._validation_results['warnings'])
            }
        
        return summary
    
    # Legacy methods for backward compatibility with strategies
    def set_lens_core_metrics(self, lens: str, **metrics):
        """Set core risk metrics for a lens."""
        if lens not in self._data:
            self._data[lens] = {'core_metrics': {}, 'matrices': {}, 'arrays': {}}
        self._data[lens]['core_metrics'].update(metrics)
    
    def set_lens_factor_exposures(self, lens: str, exposures):
        """Set factor exposures for a lens."""
        if lens not in self._data:
            self._data[lens] = {'core_metrics': {}, 'matrices': {}, 'arrays': {}}
        if hasattr(exposures, 'tolist'):
            exposures = exposures.tolist()
        self._data[lens]['arrays']['factor_exposures'] = exposures
    
    def set_lens_asset_contributions(self, lens: str, contributions):
        """Set asset contributions for a lens.""" 
        if lens not in self._data:
            self._data[lens] = {'core_metrics': {}, 'matrices': {}, 'arrays': {}}
        if hasattr(contributions, 'tolist'):
            contributions = contributions.tolist()
        self._data[lens]['arrays']['asset_contributions'] = contributions
    
    def set_lens_factor_contributions(self, lens: str, contributions):
        """Set factor contributions for a lens."""
        if lens not in self._data:
            self._data[lens] = {'core_metrics': {}, 'matrices': {}, 'arrays': {}}
        if hasattr(contributions, 'tolist'):
            contributions = contributions.tolist()
        self._data[lens]['arrays']['factor_contributions'] = contributions
    
    def set_lens_factor_risk_contributions_matrix(self, lens: str, matrix):
        """Set factor risk contributions matrix for a lens."""
        if lens not in self._data:
            self._data[lens] = {'core_metrics': {}, 'matrices': {}, 'arrays': {}}
        if hasattr(matrix, 'tolist'):
            matrix = matrix.tolist()
        self._data[lens]['matrices']['factor_risk_contributions'] = matrix
        
    def set_validation_results(self, results):
        """Set validation results."""
        self._validation_results = results
    
    def add_detail(self, key: str, value):
        """Add detail information."""
        if 'details' not in self.metadata:
            self.metadata['details'] = {}
        if hasattr(value, 'tolist'):
            value = value.tolist()
        self.metadata['details'][key] = value
    
    def add_context_info(self, key: str, value):
        """Add context information."""
        if 'context' not in self.metadata:
            self.metadata['context'] = {}
        self.metadata['context'][key] = value
    
    def __repr__(self) -> str:
        """String representation of the schema."""
        return (f"RiskResultSchema(analysis_type='{self.analysis_type}', "
                f"components={len(self._hierarchical_data)}, "
                f"time_series={len(self._time_series_data)})")