"""Main application entry point."""
import os
from flask import Flask
from src.api.routes import register_routes
from configs.logging_config import setup_logging


def create_app():
    """Create and configure the Flask application."""
    # Create the Flask application with template folder
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    
    # Configuration
    app.config['SECRET_KEY'] = 'your-secret-key'
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Flask application")
    
    # Create static/images directory if it doesn't exist
    static_images_dir = os.path.join(static_dir, 'images')
    os.makedirs(static_images_dir, exist_ok=True)
    
    # Create symlink from uploads/results/images to static/images
    uploads_images_dir = os.path.abspath(os.path.join('uploads', 'results', 'images'))
    os.makedirs(uploads_images_dir, exist_ok=True)
    
    # Create symbolic link if it doesn't exist
    try:
        # Check if the link already exists
        if not os.path.exists(os.path.join(static_images_dir, 'batch_predictions')):
            # If on Windows, use directory junction instead of symlink
            if os.name == 'nt':
                os.system(f'mklink /J "{os.path.join(static_images_dir, "batch_predictions")}" "{os.path.join(uploads_images_dir, "batch_predictions")}"')
            else:
                # Create link to batch_predictions folder
                os.symlink(os.path.join(uploads_images_dir, 'batch_predictions'), 
                           os.path.join(static_images_dir, 'batch_predictions'))
        
        if not os.path.exists(os.path.join(static_images_dir, 'calibration_curves')):
            # If on Windows, use directory junction instead of symlink
            if os.name == 'nt':
                os.system(f'mklink /J "{os.path.join(static_images_dir, "calibration_curves")}" "{os.path.join(uploads_images_dir, "calibration_curves")}"')
            else:
                # Create link to calibration_curves folder
                os.symlink(os.path.join(uploads_images_dir, 'calibration_curves'), 
                           os.path.join(static_images_dir, 'calibration_curves'))
                           
        logger.info("Static image symlinks created successfully")
    except Exception as e:
        logger.warning(f"Could not create symlinks for static images: {str(e)}")
        logger.info("Images may not display correctly in the web interface")
    
    # Register routes
    register_routes(app)
    
    return app


if __name__ == '__main__':
    app = create_app()

    app.run(debug=True)