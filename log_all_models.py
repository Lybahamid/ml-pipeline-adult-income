import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
import pandas as pd
import numpy as np

# Load and preprocess adult.test dataset
try:
    data = pd.read_csv("data/adult.test", skiprows=1, header=None)
except FileNotFoundError:
    print("Error: 'data/adult.test' not found. Please check the file path.")
    exit(1)

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
data.columns = columns
X = data.drop("income", axis=1)
y = data["income"].str.strip().map({"<=50K.": 0, ">50K.": 1})

# Preprocess categorical features
X = pd.get_dummies(X, columns=["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"])

# Define models
models = {
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss")
}

# Log each model to MLflow
mlflow.set_tracking_uri("file:///D:/ml-pipeline-adult-income/mlflow-data")
mlflow.set_experiment("adult-income-experiment")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(X, y)
        
        # Calculate metrics
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions)
        precision = precision_score(y, predictions)
        
        # Log model and metrics
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("dataset", "adult.test")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_artifact("data/adult.test")
        
        # Register the model
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
        mlflow.register_model(model_uri, model_name)