# app/main/__init__.py
from flask import Blueprint

bp = Blueprint('main', __name__)

# Import routes at the bottom to avoid circular dependencies
from . import routes