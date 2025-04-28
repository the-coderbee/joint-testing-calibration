# configs/logging_config.py

import os
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import time
from flask import request
from dotenv import load_dotenv

load_dotenv()  

# Define log directory within configs
LOG_LEVEL = os.getenv('LOG_LEVEL')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Define log files
APP_LOG = os.path.join(LOG_DIR, 'app.log')
MODEL_TRAINING_LOG = os.path.join(LOG_DIR, 'model_training.log')
MODEL_INFERENCE_LOG = os.path.join(LOG_DIR, 'model_inference.log')
API_LOG = os.path.join(LOG_DIR, 'api.log')
ERROR_LOG = os.path.join(LOG_DIR, 'error.log')

# Define custom formatter with request information
class RequestFormatter(logging.Formatter):
    def format(self, record):
        try:
            record.url = getattr(request, 'url', None)
            record.remote_addr = getattr(request, 'remote_addr', None)
        except RuntimeError:
            # Outside of Flask context
            record.url = None
            record.remote_addr = None
        return super().format(record)

# Define custom filter for model training logs
class ModelTrainingFilter(logging.Filter):
    def filter(self, record):
        return hasattr(record, 'model_training') and record.model_training

# Define custom filter for model inference
class ModelInferenceFilter(logging.Filter):
    def filter(self, record):
        return hasattr(record, 'model_inference') and record.model_inference

# Define formatters
standard_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
request_formatter = RequestFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(remote_addr)s - %(url)s - %(message)s'
)
model_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(model_name)s - %(message)s'
)

def setup_logging(app=None, log_level=LOG_LEVEL):
    """
    Set up logging for the Flask application
    
    Args:
        app: Flask application instance (optional)
        log_level: Logging level (default: INFO)
    
    Returns:
        logger: Root logger instance
    """
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(console_handler)
    
    # Main app log handler (rotating by size - 10MB max, keeping 10 backups)
    app_handler = RotatingFileHandler(APP_LOG, maxBytes=10*1024*1024, backupCount=10)
    app_handler.setLevel(log_level)
    app_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(app_handler)
    
    # Error log handler
    error_handler = TimedRotatingFileHandler(ERROR_LOG, when='midnight', interval=1, backupCount=30)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Model Training log handler (rotating daily)
    model_training_handler = TimedRotatingFileHandler(MODEL_TRAINING_LOG, when='midnight', interval=1, backupCount=30)
    model_training_handler.setLevel(log_level)
    model_training_handler.setFormatter(model_formatter)
    model_training_handler.addFilter(ModelTrainingFilter())
    
    # Model Inference log handler (rotating by size)
    model_inference_handler = RotatingFileHandler(MODEL_INFERENCE_LOG, maxBytes=20*1024*1024, backupCount=10)
    model_inference_handler.setLevel(log_level)
    model_inference_handler.setFormatter(model_formatter)
    model_inference_handler.addFilter(ModelInferenceFilter())
    
    # Create specialized loggers
    training_logger = logging.getLogger('model.training')
    training_logger.propagate = False  # Don't propagate to parent
    training_logger.addHandler(model_training_handler)
    
    inference_logger = logging.getLogger('model.inference')
    inference_logger.propagate = False  # Don't propagate to parent
    inference_logger.addHandler(model_inference_handler)
    
    # If Flask app is provided, set up request logging
    if app:
        # API request log with request context
        api_handler = RotatingFileHandler(API_LOG, maxBytes=10*1024*1024, backupCount=10)
        api_handler.setLevel(log_level)
        api_handler.setFormatter(request_formatter)
        
        # Create API logger
        api_logger = logging.getLogger('api')
        api_logger.propagate = False
        api_logger.addHandler(api_handler)
        
        # Set up Flask request logging via before_request and after_request
        @app.before_request
        def log_request_info():
            api_logger.info(f"Request: {request.method} {request.path}")
            
        @app.after_request
        def log_response_info(response):
            api_logger.info(f"Response: {response.status}")
            return response
            
        # Log unhandled exceptions
        @app.errorhandler(Exception)
        def log_exception(e):
            app.logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            return "Internal Server Error", 500
    
    return logging.getLogger()

# Function to get specific loggers
def get_training_logger(model_name):
    """Get logger for model training with model name context"""
    logger = logging.getLogger('model.training')
    extra = {'model_training': True, 'model_name': model_name}
    return logging.LoggerAdapter(logger, extra)

def get_inference_logger(model_name):
    """Get logger for model inference with model name context"""
    logger = logging.getLogger('model.inference')
    extra = {'model_inference': True, 'model_name': model_name}
    return logging.LoggerAdapter(logger, extra)

# Performance monitoring for model operations
class ModelPerformanceMonitor:
    def __init__(self, logger, operation_name):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type:
            self.logger.error(f"{self.operation_name} failed after {duration:.2f}s: {exc_val}")
        else:
            self.logger.info(f"{self.operation_name} completed in {duration:.2f}s")