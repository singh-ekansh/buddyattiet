# app/models/churn_predictor.py
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
import shap


class ChurnPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.features = None
        self.explainer = None

    def _prepare_data(self, df):
        # Convert all categorical columns to 'category' dtype for LightGBM
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')
        return df

    def train(self, df):
        df.drop('user_id', axis=1, inplace=True)
        y = df['churn']
        X = df.drop('churn', axis=1)
        self.features = X.columns.tolist()

        X_processed = self._prepare_data(X)

        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

        self.model = lgb.LGBMClassifier(objective='binary', random_state=42, n_estimators=300, learning_rate=0.05,
                                        num_leaves=40)
        print("Training LightGBM model...")
        self.model.fit(X_train, y_train)

        self.save_model()
        print("✅ Model training complete and saved.")

    def predict_with_explanation(self, input_data_df):
        if not self.model: self.load_model()

        input_df_copy = input_data_df.copy()

        # Prepare data for prediction
        input_processed = self._prepare_data(input_df_copy)
        input_processed = input_processed.reindex(columns=self.features, fill_value=0)

        # Prediction
        proba = self.model.predict_proba(input_processed)[:, 1][0]
        prediction = 1 if proba > 0.5 else 0

        # SHAP Explanation
        if self.explainer is None:
            print("Initializing SHAP explainer...")
            self.explainer = shap.TreeExplainer(self.model)

        # Create SHAP values object
        shap_values = self.explainer.shap_values(input_processed)[1]  # For the "churn" class

        explanation_plot = shap.force_plot(
            self.explainer.expected_value[1],
            shap_values[0],
            input_processed.iloc[0],
            matplotlib=False,
            show=False
        )
        # We need to get the HTML of the plot
        shap_html = f"<script>{shap.getjs()}</script>{explanation_plot.html()}"

        return prediction, proba, shap_html

    def save_model(self):
        payload = {'model': self.model, 'features': self.features}
        joblib.dump(payload, self.model_path)

    def load_model(self):
        if self.model is None:
            payload = joblib.load(self.model_path)
            self.model = payload['model']
            self.features = payload['features']
            print("🧠 Model loaded from disk.")