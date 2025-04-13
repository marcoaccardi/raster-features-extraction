#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging configuration for the raster feature extraction pipeline.

This module provides centralized configuration for the logging system
used throughout the application.
"""
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Dict, Optional
from raster_features.core.config import LOGGING_CONFIG, DEFAULT_OUTPUT_DIR

def setup_logging(log_level: Optional[str] = None, 
                  log_file: Optional[str] = None,
                  module_name: str = "feature_extraction") -> logging.Logger:
    """
    Configure and return a logger with the specified settings.
    
    Parameters
    ----------
    log_level : str, optional
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        If None, uses the level from config.py.
    log_file : str, optional
        Path to log file. If None, uses the path from config.py.
    module_name : str, optional
        Name of the module for the logger, by default "feature_extraction".
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(module_name)
    
    # If logger is already configured, return it
    if logger.hasHandlers():
        return logger
    
    # Get configuration
    level = log_level or LOGGING_CONFIG.get("level", "INFO")
    log_to_file = LOGGING_CONFIG.get("log_to_file", True)
    log_file_path = log_file or LOGGING_CONFIG.get("log_file")
    log_format = LOGGING_CONFIG.get("log_format", 
                                   "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file and log_file_path:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized at level: {level}")
    return logger

def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Parameters
    ----------
    module_name : str
        Name of the module, typically __name__.
        
    Returns
    -------
    logging.Logger
        Configured logger for the module.
    """
    logger_name = f"feature_extraction.{module_name}"
    return logging.getLogger(logger_name)

# Initialize the root logger
root_logger = setup_logging()
