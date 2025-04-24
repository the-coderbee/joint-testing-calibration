"""Main application entry point."""
import os
from flask import Flask
from src.api.routes import register_routes


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
    
    # Register routes
    register_routes(app)
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True) 