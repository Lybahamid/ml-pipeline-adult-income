## Adult Income ML Pipeline
This project implements a machine learning pipeline for predicting income levels (>50K or <=50K) using the UCI Adult Income dataset. It includes data ingestion, preprocessing, training of three models (RandomForestClassifier, LogisticRegression, XGBoost), evaluation, MLflow logging, and a FastAPI-based prediction API deployed with Docker.

Dataset Source and Description

Source: UCI Machine Learning Repository - Adult Dataset (link).
Description: The dataset has 48,842 instances with 14 features (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country) and a binary target (income: <=50K or >50K).
Example Dataset: The adult.test file (16,281 instances) is used for testing. A sample test.csv is included for API testing. Download the full dataset from the source link.

Steps to Run the Pipeline

Setup Environment:

Create and activate a Conda environment:conda create -n lfw_env python=3.10
conda activate lfw_env
pip install -r requirements.txt




Run the Pipeline:

Execute main.py to ingest data, train models, evaluate, and save best_model_XGBoost.joblib:python main.py


(Optional) Run log_all_models.py to ensure all three models are logged to MLflow:python log_all_models.py




Run MLflow UI:

Run with Docker Compose (preferred, see Step 6) or directly:docker run -p 5000:5000 -v D:/ml-pipeline-adult-income/mlflow-data:/mlflow-data my-mlflow-app mlflow ui --backend-store-uri file:///mlflow-data --host 0.0.0.0 --static-prefix /static-files


Access at http://localhost:5000/static-files.


Run Prediction API:

Build the API image:docker build -t ml-pipeline .


Run:docker run -p 8000:8000 ml-pipeline


Test with:curl -X POST http://localhost:8000/predict -F "file=@test.csv"




Log API Predictions:

Run:python evaluate_api.py




Run with Docker Compose:

Use docker-compose.yml to run both MLflow UI and API:docker-compose up


Access MLflow UI at http://localhost:5000/static-files and API at http://localhost:8000/predict.



Models Tested and Final Results
Three models were trained and evaluated on adult.test:

RandomForestClassifier:
Accuracy: [0.9999385787113814]
F1 Score: [0.9998699778962423]
Precision: [1]


LogisticRegression:
Accuracy: [0.798968122351207]
F1 Score: [0.37093984239861616]
Precision: [0.7111274871039057]

XGBoost:
Accuracy: [0.915484306860758]
F1 Score: [0.8078212290502792]
Precision: [0.8726614363307181]


API Predictions:
Accuracy: [0.6572377158034528]
F1 Score: [0.22655079412646087]
Precision: [0.2441860465116279]


The RandomForestClassifier model outperformed others and is used in the API (best_model.joblib).

License:

MIT License

