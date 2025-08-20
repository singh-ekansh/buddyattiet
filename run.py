# run.py
import os
from app import create_app
from app.data_generator import generate_ott_data
from app.models.churn_predictor import ChurnPredictor
import pandas as pd

# --- Centralized Configuration ---
# Define paths at the top level of the entrypoint script
# This is the single source of truth for file locations.
ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
DATA_PATH = os.path.join(DATA_DIR, 'india_ott_market_data.csv')
MODEL_PATH = os.path.join(DATA_DIR, 'churn_model.joblib')


def setup_environment():
    """Ensures data and model exist, creating them if necessary."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if the data file exists and is not empty
    if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
        if os.path.exists(DATA_PATH):
            print("Data file is empty. Regenerating...")
        else:
            print("Data file not found. Generating...")

        df = generate_ott_data()
        df.to_csv(DATA_PATH, index=False)
        print(f"✅ Data generated and saved to {DATA_PATH}")

    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training a new one...")
        df = pd.read_csv(DATA_PATH)
        predictor = ChurnPredictor(model_path=MODEL_PATH)
        predictor.train(df)


# --- Main Application Execution ---
if __name__ == '__main__':
    # 1. CRITICAL: Run setup BEFORE creating the app.
    # This guarantees the data file exists and is populated.
    setup_environment()

    # 2. Create the Flask app instance by passing the config paths.
    app = create_app(config={
        'DATA_PATH': DATA_PATH,
        'MODEL_PATH': MODEL_PATH
    })

    # 3. Run the application
    print("🚀 Starting Enterprise Churn Intelligence Platform...")
    app.run(debug=True, port=5001)