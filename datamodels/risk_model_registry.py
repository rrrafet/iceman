"""
Risk Model Registry for portfolio risk analysis system.
Manages risk model metadata, registration, and validation.
"""

from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RiskModelRegistry:
    """
    Registry for managing risk model metadata and configuration.
    
    Provides centralized management of available risk models with metadata,
    validation, and configuration capabilities.
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize risk model registry.
        
        Args:
            registry_path: Optional path to save/load registry data
        """
        self.registry_path = Path(registry_path) if registry_path else None
        self._models: Dict[str, Dict[str, Any]] = {}
        self._current_model: Optional[str] = None
        
        if self.registry_path and self.registry_path.exists():
            self._load_registry()
    
    def register_model(self, code: str, metadata: Dict[str, Any]) -> bool:
        """
        Register a risk model with metadata.
        
        Args:
            code: Unique code for the risk model
            metadata: Dictionary containing model metadata
            
        Returns:
            True if registration successful, False otherwise
        """
        if not code or not isinstance(metadata, dict):
            logger.error(f"Invalid parameters for model registration: code='{code}', metadata type={type(metadata)}")
            return False
        
        # Validate required metadata fields
        required_fields = {'name', 'description', 'factors'}
        missing_fields = required_fields - set(metadata.keys())
        if missing_fields:
            logger.error(f"Missing required metadata fields for model {code}: {missing_fields}")
            return False
        
        # Add registration timestamp
        metadata_with_timestamp = metadata.copy()
        metadata_with_timestamp['registered_at'] = datetime.now().isoformat()
        metadata_with_timestamp['code'] = code
        
        # Validate factors list
        if not isinstance(metadata['factors'], list):
            logger.error(f"Factors must be a list for model {code}")
            return False
        
        self._models[code] = metadata_with_timestamp
        
        # Set as current model if it's the first one
        if self._current_model is None:
            self._current_model = code
        
        logger.info(f"Registered risk model '{code}' with {len(metadata['factors'])} factors")
        
        # Save registry if path is configured
        if self.registry_path:
            self._save_registry()
        
        return True
    
    def get_model_info(self, code: str) -> Dict[str, Any]:
        """
        Get metadata for a specific risk model.
        
        Args:
            code: Risk model code
            
        Returns:
            Dictionary containing model metadata
        """
        if code not in self._models:
            logger.warning(f"Risk model not found: {code}")
            return {}
        
        return self._models[code].copy()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get list of all available models with their metadata.
        
        Returns:
            List of dictionaries containing model information
        """
        models_list = []
        for code, metadata in self._models.items():
            model_info = metadata.copy()
            model_info['is_current'] = code == self._current_model
            models_list.append(model_info)
        
        # Sort by registration date (newest first)
        models_list.sort(key=lambda x: x.get('registered_at', ''), reverse=True)
        
        logger.debug(f"Listed {len(models_list)} available risk models")
        return models_list
    
    def get_current_model(self) -> str:
        """
        Get currently selected risk model code.
        
        Returns:
            Current risk model code or empty string if none selected
        """
        return self._current_model or ""
    
    def set_current_model(self, code: str) -> bool:
        """
        Set the current active risk model.
        
        Args:
            code: Risk model code to set as current
            
        Returns:
            True if successful, False if model not found
        """
        if code not in self._models:
            logger.error(f"Cannot set current model - risk model not found: {code}")
            return False
        
        previous_model = self._current_model
        self._current_model = code
        
        logger.info(f"Changed current risk model from '{previous_model}' to '{code}'")
        
        # Save registry if path is configured
        if self.registry_path:
            self._save_registry()
        
        return True
    
    def get_model_factors(self, code: str) -> List[str]:
        """
        Get list of factors for a specific risk model.
        
        Args:
            code: Risk model code
            
        Returns:
            List of factor names
        """
        model_info = self.get_model_info(code)
        factors = model_info.get('factors', [])
        
        logger.debug(f"Retrieved {len(factors)} factors for model {code}")
        return factors
    
    def validate_model_data(self, code: str, factor_data_provider=None) -> Dict[str, Any]:
        """
        Validate model data completeness and consistency.
        
        Args:
            code: Risk model code to validate
            factor_data_provider: Optional FactorDataProvider for data validation
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "model_code": code,
            "valid": False,
            "errors": [],
            "warnings": [],
            "metadata_valid": False,
            "data_valid": False
        }
        
        # Validate model registration
        if code not in self._models:
            validation_result["errors"].append(f"Model not registered: {code}")
            return validation_result
        
        model_info = self._models[code]
        
        # Validate metadata completeness
        required_fields = {'name', 'description', 'factors', 'registered_at'}
        missing_fields = required_fields - set(model_info.keys())
        if missing_fields:
            validation_result["errors"].append(f"Missing metadata fields: {missing_fields}")
        else:
            validation_result["metadata_valid"] = True
        
        # Validate factors list
        factors = model_info.get('factors', [])
        if not factors:
            validation_result["errors"].append("No factors defined for model")
        elif not isinstance(factors, list):
            validation_result["errors"].append("Factors must be a list")
        else:
            # Check for duplicate factors
            if len(factors) != len(set(factors)):
                validation_result["warnings"].append("Duplicate factors found")
        
        # Validate against actual data if provider is available
        if factor_data_provider and validation_result["metadata_valid"]:
            try:
                # Check if model exists in data
                available_models = factor_data_provider.get_available_risk_models()
                if code not in available_models:
                    validation_result["errors"].append(f"Model not found in factor data: {code}")
                else:
                    # Validate factor completeness
                    data_factors = set(factor_data_provider.get_factor_names(code))
                    registered_factors = set(factors)
                    
                    missing_in_data = registered_factors - data_factors
                    extra_in_data = data_factors - registered_factors
                    
                    if missing_in_data:
                        validation_result["errors"].append(f"Factors missing in data: {missing_in_data}")
                    
                    if extra_in_data:
                        validation_result["warnings"].append(f"Extra factors in data: {extra_in_data}")
                    
                    if not missing_in_data:
                        validation_result["data_valid"] = True
                        
                        # Additional data quality checks
                        data_validation = factor_data_provider.validate_data_completeness(code)
                        if not data_validation.get("valid", False):
                            validation_result["warnings"].append("Data quality issues detected")
                        
            except Exception as e:
                validation_result["errors"].append(f"Data validation error: {str(e)}")
        
        # Overall validation status
        validation_result["valid"] = (
            validation_result["metadata_valid"] and 
            len(validation_result["errors"]) == 0
        )
        
        logger.info(f"Model validation for {code}: {'PASS' if validation_result['valid'] else 'FAIL'}")
        return validation_result
    
    def get_model_summary(self, code: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of a risk model.
        
        Args:
            code: Risk model code
            
        Returns:
            Dictionary with model summary information
        """
        if code not in self._models:
            return {"error": f"Model not found: {code}"}
        
        model_info = self._models[code]
        
        summary = {
            "code": code,
            "name": model_info.get("name", "Unknown"),
            "description": model_info.get("description", "No description"),
            "factor_count": len(model_info.get("factors", [])),
            "factors": model_info.get("factors", []),
            "registered_at": model_info.get("registered_at", "Unknown"),
            "is_current": code == self._current_model,
            "metadata": {k: v for k, v in model_info.items() if k not in ['factors', 'code']}
        }
        
        return summary
    
    def remove_model(self, code: str) -> bool:
        """
        Remove a risk model from the registry.
        
        Args:
            code: Risk model code to remove
            
        Returns:
            True if successful, False if model not found
        """
        if code not in self._models:
            logger.warning(f"Cannot remove model - not found: {code}")
            return False
        
        del self._models[code]
        
        # Update current model if it was removed
        if self._current_model == code:
            remaining_models = list(self._models.keys())
            self._current_model = remaining_models[0] if remaining_models else None
            logger.info(f"Removed current model {code}, new current model: {self._current_model}")
        
        logger.info(f"Removed risk model: {code}")
        
        # Save registry if path is configured
        if self.registry_path:
            self._save_registry()
        
        return True
    
    def _load_registry(self) -> None:
        """Load registry data from file."""
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
            
            self._models = data.get('models', {})
            self._current_model = data.get('current_model')
            
            logger.info(f"Loaded risk model registry with {len(self._models)} models")
            
        except Exception as e:
            logger.warning(f"Failed to load registry from {self.registry_path}: {e}")
            self._models = {}
            self._current_model = None
    
    def _save_registry(self) -> None:
        """Save registry data to file."""
        if not self.registry_path:
            return
        
        try:
            # Ensure directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'models': self._models,
                'current_model': self._current_model,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved risk model registry to {self.registry_path}")
            
        except Exception as e:
            logger.error(f"Failed to save registry to {self.registry_path}: {e}")
    
    def clear_registry(self) -> None:
        """Clear all registered models."""
        self._models.clear()
        self._current_model = None
        
        if self.registry_path:
            self._save_registry()
        
        logger.info("Cleared all registered risk models")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        total_factors = sum(len(model.get('factors', [])) for model in self._models.values())
        
        stats = {
            "total_models": len(self._models),
            "current_model": self._current_model,
            "total_factors": total_factors,
            "average_factors_per_model": total_factors / len(self._models) if self._models else 0,
            "registry_path": str(self.registry_path) if self.registry_path else None,
            "models": list(self._models.keys())
        }
        
        return stats