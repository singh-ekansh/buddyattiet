# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv('ott_churn_data.csv')

# Define features (X) and target (y)
X = df.drop(['churn', 'user_id'], axis=1)
y = df['churn']

# Identify categorical and numerical features
categorical_features = ['gender', 'location_city', 'plan_type', 'payment_method', 'auto_renewal']
numerical_features = X.columns.drop(categorical_features)

# Create a preprocessor object using ColumnTransformer
# This is the best practice for handling different data types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create the full model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, class_weight='balanced'))
])

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save the entire pipeline (preprocessor + model) to a file
joblib.dump(model, 'churn_model.joblib')

print("Model training complete and 'churn_model.joblib' saved.")