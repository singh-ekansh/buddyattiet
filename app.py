import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import plotly.express as px
import plotly.io as pio
import json

# Initialize the Flask application
app = Flask(__name__)

# --- Global variable to hold the trained model ---
model = None

def train_model():
    """
    Loads data, preprocesses it, and trains a Random Forest model.
    """
    global model

    try:
        df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')
    except Exception as e:
        print(f"Could not load data: {e}")
        return None

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(columns=['customerID'], inplace=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])
    model.fit(X, y)
    print("Model trained successfully!")

def analyze_churn_data():
    """
    This function loads and analyzes data for the dashboard visualizations using Plotly.
    """
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')
    except Exception as e:
        print(f"Could not load data: {e}")
        return None, None

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn_numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    churn_rate = round(df['Churn_numeric'].mean() * 100, 1)

    # --- Create Plotly Visualizations ---
    layout_template = {
        'font': {'family': 'Inter, sans-serif', 'color': '#374151'},
        'paper_bgcolor': 'white',
        'plot_bgcolor': 'white',
        'title': {'x': 0.5, 'font': {'size': 20, 'family': 'Inter, sans-serif'}},
        'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': -0.4, 'xanchor': 'center', 'x': 0.5}
    }

    churn_dist_df = df['Churn'].value_counts().reset_index()
    fig_dist = px.pie(churn_dist_df, values='count', names='Churn',
                      color='Churn',
                      color_discrete_map={'No': 'rgba(59, 130, 246, 0.8)', 'Yes': 'rgba(239, 68, 68, 0.8)'},
                      hole=.4)
    fig_dist.update_layout(layout_template, showlegend=True, title_text='Churn Distribution')
    fig_dist.update_traces(textinfo='percent', textfont_size=14, marker=dict(line=dict(color='#FFFFFF', width=2)))

    contract_df = df.groupby('Contract')['Churn_numeric'].mean().mul(100).round(1).reset_index()
    fig_contract = px.bar(contract_df, x='Churn_numeric', y='Contract', orientation='h',
                          color='Contract',
                          color_discrete_sequence=['rgba(239, 68, 68, 0.7)', 'rgba(251, 146, 60, 0.7)', 'rgba(59, 130, 246, 0.7)'])
    fig_contract.update_layout(layout_template, showlegend=False, title_text='Churn by Contract Type', xaxis_title='Churn Rate (%)', yaxis_title=None, xaxis_range=[0,50])

    dependents_df = df.groupby('Dependents')['Churn_numeric'].mean().mul(100).round(1).reset_index()
    fig_dependents = px.bar(dependents_df, x='Dependents', y='Churn_numeric',
                            color='Dependents',
                            color_discrete_map={'No': 'rgba(239, 68, 68, 0.7)', 'Yes': 'rgba(59, 130, 246, 0.7)'})
    fig_dependents.update_layout(layout_template, showlegend=False, title_text='Churn by Dependents', xaxis_title=None, yaxis_title='Churn Rate (%)', yaxis_range=[0,40])

    internet_df = df.groupby('InternetService')['Churn_numeric'].mean().mul(100).round(1).reset_index()
    fig_internet = px.bar(internet_df, x='InternetService', y='Churn_numeric',
                          color='InternetService',
                          color_discrete_map={'DSL': 'rgba(251, 146, 60, 0.7)', 'Fiber optic': 'rgba(239, 68, 68, 0.7)', 'No': 'rgba(59, 130, 246, 0.7)'})
    fig_internet.update_layout(layout_template, showlegend=False, title_text='Churn by Internet Service', xaxis_title=None, yaxis_title='Churn Rate (%)', yaxis_range=[0,50])

    chart_data_json = {
        'churn_distribution': pio.to_json(fig_dist),
        'churn_by_contract': pio.to_json(fig_contract),
        'churn_by_dependents': pio.to_json(fig_dependents),
        'churn_by_internet_service': pio.to_json(fig_internet)
    }
    return churn_rate, chart_data_json

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    churn_rate, chart_data_json = analyze_churn_data()
    prediction_text = ""
    prediction_proba = 0

    if request.method == 'POST':
        form_data = request.form.to_dict()
        form_data['tenure'] = int(form_data['tenure'])
        form_data['MonthlyCharges'] = float(form_data['MonthlyCharges'])
        form_data['TotalCharges'] = float(form_data['TotalCharges'])
        form_data['SeniorCitizen'] = int(form_data['SeniorCitizen'])
        input_df = pd.DataFrame([form_data])

        if model:
            proba = model.predict_proba(input_df)[0][1]
            prediction_proba = round(proba * 100)
            prediction_text = "High risk of churn." if prediction_proba > 50 else "Low risk of churn."

    if churn_rate is None:
        return "Error: Could not process data.", 500

    return render_template('_base.html',
                           churn_rate=churn_rate,
                           chart_data_json=chart_data_json,
                           prediction_text=prediction_text,
                           prediction_proba=prediction_proba)

if __name__ == '__main__':
    train_model()

    app.run(debug=True)
