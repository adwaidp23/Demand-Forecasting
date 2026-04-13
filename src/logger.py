import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name="demand_forecasting", log_file="logs/app.log"):
    """Sets up a centralized logger with console and file handlers."""
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File Handler
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    
    # Add handlers if not already present
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger

# Singleton instance
logger = setup_logger()
