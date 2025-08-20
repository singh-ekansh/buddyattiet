# app/__init__.py
import os
from flask import Flask

# --- Absolute Path Configuration ---
# Get the absolute path of the directory containing this __init__.py file.
# This is the .../jiohotstar-churn-analysis/app/ directory.
APP_DIR = os.path.abspath(os.path.dirname(__file__))

def create_app(config=None):
    """
    This is the application factory. It creates and configures the Flask app.
    """
    # Use the absolute paths to remove any ambiguity for Flask.
    app = Flask(__name__,
                template_folder=os.path.join(APP_DIR, 'templates'),
                static_folder=os.path.join(APP_DIR, 'static')
               )

    # Load the configuration passed from run.py
    if config:
        app.config.update(config)

    # Import and register blueprints
    from .main import bp as main_blueprint
    from .api import bp as api_blueprint

    app.register_blueprint(main_blueprint)
    app.register_blueprint(api_blueprint, url_prefix='/api')

    return app