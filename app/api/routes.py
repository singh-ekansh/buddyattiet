# app/api/routes.py
from flask import request, jsonify, current_app
from . import bp
from app.models.churn_predictor import ChurnPredictor
import pandas as pd

# --- Lazy Initialization ---
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        model_path = current_app.config['MODEL_PATH']
        predictor = ChurnPredictor(model_path=model_path)
    return predictor


@bp.route('/predict', methods=['POST'])
def predict():
    pred_service = get_predictor()
    data = request.get_json()

    # Convert numeric fields
    for key in ['age', 'monthly_price', 'tenure_months', 'monthly_watch_hours', 'days_since_last_login',
                'support_tickets', 'watched_genres_count']:
        data[key] = pd.to_numeric(data[key])

    input_df = pd.DataFrame([data])

    prediction, probability, shap_html = pred_service.predict_with_explanation(input_df)

    return jsonify({
        'prediction': int(prediction),
        'probability': float(probability),
        'shap_html': shap_html
    })