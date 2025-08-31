"""
Configuration Service for portfolio risk analysis system.
Manages YAML configuration loading, settings, and defaults.
"""

from typing import Dict, Any, List, Optional
import yaml
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigurationService:
    """
    Manages configuration loading and settings for the risk analysis system.
    
    Provides centralized configuration management with YAML loading,
    default settings, and dynamic configuration updates.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration service.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self._config: Dict[str, Any] = {}
        self._defaults = self._get_default_config()
        
        if self.config_path and self.config_path.exists():
            self.load_config(str(self.config_path))
        else:
            logger.info("Using default configuration")
            self._config = self._defaults.copy()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration structure."""
        return {
            "risk_model": {
                "default": "macro1",
                "available": ["macro1"]
            },
            "portfolio": {
                "default": "strategic_portfolio",
                "root_component_id": "TOTAL"
            },
            "analysis": {
                "annualized": True,
                "frequency": "daily",
                "currency": "USD",
                "risk_model_type": "factor_based"
            },
            "ui_defaults": {
                "lens": "portfolio",
                "show_top_n_factors": 8,
                "show_top_n_assets": 8,
                "default_date_range": "latest_year"
            },
            "data_sources": {
                "portfolio_data": "data/portfolio.parquet",
                "factor_returns": "data/factor_returns.parquet"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        try:
            config_path_obj = Path(config_path)
            
            if not config_path_obj.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return self._defaults.copy()
            
            with open(config_path_obj, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
            
            if not loaded_config:
                logger.warning(f"Empty configuration file: {config_path}")
                return self._defaults.copy()
            
            # Merge with defaults (loaded config takes precedence)
            self._config = self._merge_configs(self._defaults, loaded_config)
            self.config_path = config_path_obj
            
            logger.info(f"Loaded configuration from {config_path}")
            return self._config.copy()
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {config_path}: {e}")
            return self._defaults.copy()
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            return self._defaults.copy()
    
    def _merge_configs(self, defaults: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge loaded config with defaults.
        
        Args:
            defaults: Default configuration
            loaded: Loaded configuration
            
        Returns:
            Merged configuration
        """
        merged = defaults.copy()
        
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get_default_risk_model(self) -> str:
        """
        Get default risk model code.
        
        Returns:
            Default risk model code
        """
        return self._config.get("risk_model", {}).get("default", "macro1")
    
    def get_available_risk_models(self) -> List[str]:
        """
        Get list of available risk models from configuration.
        
        Returns:
            List of available risk model codes
        """
        return self._config.get("risk_model", {}).get("available", ["macro1"])
    
    def get_default_portfolio(self) -> str:
        """
        Get default portfolio name.
        
        Returns:
            Default portfolio name
        """
        return self._config.get("portfolio", {}).get("default", "strategic_portfolio")
    
    def get_root_component_id(self) -> str:
        """
        Get root component ID for portfolio hierarchy.
        
        Returns:
            Root component ID
        """
        return self._config.get("portfolio", {}).get("root_component_id", "TOTAL")
    
    def get_analysis_settings(self) -> Dict[str, Any]:
        """
        Get analysis settings.
        
        Returns:
            Dictionary with analysis settings (annualized, currency, etc.)
        """
        return self._config.get("analysis", {}).copy()
    
    def get_ui_settings(self) -> Dict[str, Any]:
        """
        Get UI default settings.
        
        Returns:
            Dictionary with UI settings (default lens, date range, etc.)
        """
        return self._config.get("ui_defaults", {}).copy()
    
    def get_data_sources(self) -> Dict[str, str]:
        """
        Get data source paths.
        
        Returns:
            Dictionary with data source file paths
        """
        return self._config.get("data_sources", {}).copy()
    
    def update_setting(self, key: str, value: Any) -> bool:
        """
        Update a configuration setting using dot notation.
        
        Args:
            key: Setting key (supports dot notation like 'risk_model.default')
            value: New value
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            keys = key.split('.')
            current = self._config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the final value
            current[keys[-1]] = value
            
            logger.info(f"Updated configuration: {key} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update setting {key}: {e}")
            return False
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration setting using dot notation.
        
        Args:
            key: Setting key (supports dot notation like 'risk_model.default')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key.split('.')
            current = self._config
            
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            
            return current
            
        except Exception as e:
            logger.error(f"Error getting setting {key}: {e}")
            return default
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to YAML file.
        
        Args:
            config_path: Optional path to save to (uses loaded path if not provided)
            
        Returns:
            True if save successful, False otherwise
        """
        save_path = Path(config_path) if config_path else self.config_path
        
        if not save_path:
            logger.error("No save path specified and no config file loaded")
            return False
        
        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            config_to_save = self._config.copy()
            config_to_save["_metadata"] = {
                "saved_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {save_path}: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = self._defaults.copy()
        logger.info("Reset configuration to defaults")
    
    def get_factor_subset(self) -> Optional[List[str]]:
        """
        Get factor subset if configured (for filtering factors).
        
        Returns:
            List of factor names or None for all factors
        """
        return self.get_setting("analysis.factor_subset", None)
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate current configuration.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required sections
        required_sections = ["risk_model", "portfolio", "analysis", "ui_defaults"]
        for section in required_sections:
            if section not in self._config:
                validation_result["errors"].append(f"Missing required section: {section}")
        
        # Validate risk model settings
        risk_model_config = self._config.get("risk_model", {})
        if "default" not in risk_model_config:
            validation_result["errors"].append("Missing risk_model.default")
        
        available_models = risk_model_config.get("available", [])
        default_model = risk_model_config.get("default")
        if default_model and default_model not in available_models:
            validation_result["warnings"].append(
                f"Default risk model '{default_model}' not in available models"
            )
        
        # Validate data sources
        data_sources = self._config.get("data_sources", {})
        for source_name, source_path in data_sources.items():
            if source_path and not Path(source_path).exists():
                validation_result["warnings"].append(f"Data source file not found: {source_path}")
        
        # Validate analysis settings
        analysis_config = self._config.get("analysis", {})
        if analysis_config.get("frequency") not in ["daily", "weekly", "monthly", None]:
            validation_result["warnings"].append("Invalid analysis frequency")
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        logger.info(f"Configuration validation: {'PASS' if validation_result['valid'] else 'FAIL'}")
        return validation_result
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive configuration summary.
        
        Returns:
            Dictionary with configuration summary
        """
        summary = {
            "config_path": str(self.config_path) if self.config_path else None,
            "sections": list(self._config.keys()),
            "default_risk_model": self.get_default_risk_model(),
            "available_risk_models": self.get_available_risk_models(),
            "default_portfolio": self.get_default_portfolio(),
            "root_component_id": self.get_root_component_id(),
            "analysis_settings": self.get_analysis_settings(),
            "ui_settings": self.get_ui_settings(),
            "data_sources": self.get_data_sources()
        }
        
        # Add validation status
        validation = self.validate_config()
        summary["validation"] = {
            "valid": validation["valid"],
            "error_count": len(validation["errors"]),
            "warning_count": len(validation["warnings"])
        }
        
        return summary
    
    def get_full_config(self) -> Dict[str, Any]:
        """
        Get complete configuration dictionary.
        
        Returns:
            Complete configuration
        """
        return self._config.copy()
    
    def update_from_dict(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration from a dictionary.
        
        Args:
            updates: Dictionary with configuration updates
            
        Returns:
            True if update successful
        """
        try:
            self._config = self._merge_configs(self._config, updates)
            logger.info(f"Updated configuration with {len(updates)} changes")
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration from dict: {e}")
            return False