"""
Spark Package Logging Configuration
===================================

Centralized logging configuration for the Spark package.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def configure_spark_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
    file_handler: Optional[str] = None
) -> None:
    """
    Configure logging for the Spark package.
    
    Parameters
    ----------
    level : str, default "INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format_string : str, optional
        Custom format string for log messages
    include_timestamp : bool, default True
        Whether to include timestamp in log messages
    file_handler : str, optional
        Path to log file for file output
    """
    # Default format string
    if format_string is None:
        if include_timestamp:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            format_string = "%(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger for spark package
    logger = logging.getLogger("spark")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if file_handler:
        file_handler_obj = logging.FileHandler(file_handler)
        file_handler_obj.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_string)
        file_handler_obj.setFormatter(file_formatter)
        logger.addHandler(file_handler_obj)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module within the spark package.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    return logging.getLogger(f"spark.{name}")


# Pre-configured logging setups
def setup_development_logging():
    """Setup logging configuration for development environment."""
    configure_spark_logging(
        level="DEBUG",
        include_timestamp=True,
        format_string="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )


def setup_production_logging(log_file: str = "spark.log"):
    """Setup logging configuration for production environment."""
    configure_spark_logging(
        level="INFO", 
        include_timestamp=True,
        file_handler=log_file
    )


def setup_testing_logging():
    """Setup minimal logging configuration for testing."""
    configure_spark_logging(
        level="WARNING",
        include_timestamp=False,
        format_string="%(name)s - %(levelname)s - %(message)s"
    )


def setup_risk_decomposition_logging(
    log_level: str = "INFO",
    verbose: bool = False,
    log_file: Optional[str] = None,
    overwrite: bool = True
) -> logging.Logger:
    """
    Setup specialized logging for risk decomposition visitors.
    
    Parameters
    ----------
    log_level : str, default "INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    verbose : bool, default False
        Enable console output alongside file logging
    log_file : str, optional
        Custom log file path. If None, generates timestamped file in spark/logs/
    overwrite : bool, default True
        Whether to overwrite existing log file
        
    Returns
    -------
    logging.Logger
        Configured logger for risk decomposition
    """
    # Generate default log file path if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / f"factor_risk_decomposition_{timestamp}.log")
    
    # Create logger
    logger_name = "spark.portfolio.risk_decomposition"
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Detailed format for risk decomposition logging
    detailed_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    formatter = logging.Formatter(detailed_format)
    
    # File handler (always present)
    file_mode = 'w' if overwrite else 'a'
    file_handler = logging.FileHandler(log_file, mode=file_mode)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (only if verbose)
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    # Log initialization message
    logger.info(f"Risk decomposition logging initialized: level={log_level}, verbose={verbose}, file={log_file}")
    
    return logger


def get_risk_decomposition_logger() -> logging.Logger:
    """
    Get the existing risk decomposition logger or create a default one.
    
    Returns
    -------
    logging.Logger
        Risk decomposition logger instance
    """
    logger_name = "spark.portfolio.risk_decomposition"
    logger = logging.getLogger(logger_name)
    
    # If logger has no handlers, set up default configuration
    if not logger.handlers:
        return setup_risk_decomposition_logging()
    
    return logger