import pandas as pd
from data_ingestion import ingest_data
from train import train_models
from evaluate import evaluate_model
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.debug("Starting pipeline")
    # Ingest
    logging.debug("Running data ingestion")
    df = ingest_data(os.path.join('data', 'adult.data'))
    
    # Train
    logging.debug("Running model training")
    model, X_test, y_test = train_models(df)
    
    # Evaluate
    logging.debug("Running model evaluation")
    evaluate_model(model, X_test, y_test)
    
    logging.debug("Pipeline complete")
    print("Pipeline complete. Check artifacts/ for outputs.")