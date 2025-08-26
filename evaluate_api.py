import requests
from sklearn.metrics import accuracy_score, f1_score, precision_score
import mlflow
import pandas as pd
import io
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

logging.debug("Starting evaluate_api.py")
try:
    logging.debug("Loading data/adult.test")
    data = pd.read_csv("data/adult.test", skiprows=1, header=None)
except FileNotFoundError:
    logging.error("Error: 'data/adult.test' not found. Please check the file path.")
    exit(1)

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
data.columns = columns
true_labels = data["income"].str.strip().map({"<=50K.": 0, ">50K.": 1})
logging.debug("Data loaded and labels prepared")

# Prepare CSV content
csv_buffer = io.StringIO()
data.to_csv(csv_buffer, index=False)
csv_content = csv_buffer.getvalue().encode('utf-8')
logging.debug("CSV content prepared")

# Send CSV to API
try:
    logging.debug("Sending request to API")
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("adult.test", csv_content, "text/csv")}
    )
    response.raise_for_status()
    predictions = response.json().get("predictions")
    logging.debug("Received API response: %s", predictions)
except requests.exceptions.RequestException as e:
    logging.error(f"Error with API request: {e}")
    exit(1)

# Filter valid predictions
valid_indices = [i for i, p in enumerate(predictions) if p is not None]
true_labels = true_labels.iloc[valid_indices]
predictions = [p for p in predictions if p is not None]
logging.debug("Filtered valid predictions")

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
logging.debug("Metrics calculated - Accuracy: %s, F1: %s, Precision: %s", accuracy, f1, precision)

# Log to MLflow
logging.debug("Setting MLflow tracking URI")
mlflow.set_tracking_uri("file:///D:/ml-pipeline-adult-income/mlflow-data")  # Or "sqlite:///D:/ml-pipeline-adult-income/mlflow-data/mlflow.db"
mlflow.set_experiment("adult-income-experiment")
with mlflow.start_run(run_name="API_Predictions"):
    mlflow.log_metric("api_accuracy", accuracy)
    mlflow.log_metric("api_f1_score", f1)
    mlflow.log_metric("api_precision", precision)
    logging.debug("Metrics logged to MLflow")