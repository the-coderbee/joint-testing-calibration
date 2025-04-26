import os
import sys
from configs.logging_config import setup_logging, get_training_logger, ModelPerformanceMonitor
from src.ml.train import train_model


if __name__ == "__main__":
    setup_logging()
    
    model_name = "calibration_model"
    logger = get_training_logger(model_name)
    logger.info("Starting model training script")
    
    try:
        # Set project root to current directory
        project_root = os.path.abspath(os.path.dirname(__file__))
        
        data_path = os.path.join(project_root, 'data', 'raw_data', '*.csv')
        output_path = os.path.join(project_root, 'calibration_pipeline.joblib')
        
        with ModelPerformanceMonitor(logger, "Model Training"):
            train_model(data_path, output_path, logger=logger)
        
        logger.info("Model training completed successfully")
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        sys.exit(1)
