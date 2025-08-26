import joblib
import pandas as pd
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def make_predictions(model_path, new_data_path):
    """
    Load model and predict on new data (adult.test).
    """
    logging.debug("Starting make_predictions")
    try:
        # Load model
        logging.debug(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Load new data, skipping first line
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income'
        ]
        logging.debug(f"Loading test data from {new_data_path}")
        new_df = pd.read_csv(new_data_path, names=columns, skipinitialspace=True, skiprows=1)
        
        # Clean income labels (remove trailing periods)
        if 'income' in new_df.columns:
            new_df['income'] = new_df['income'].str.replace(r'\.$', '', regex=True)
            logging.debug("Cleaned income labels")
        
        # Drop fnlwgt and income (if present) to match training data
        new_df = new_df.drop(['fnlwgt', 'income'], axis=1, errors='ignore')
        logging.debug("Dropped fnlwgt and income columns")
        
        # Drop rows with missing values ('?') to match training preprocessing
        new_df = new_df.replace('?', pd.NA)
        pre_drop_rows = len(new_df)
        new_df = new_df.dropna()
        logging.debug(f"Dropped {pre_drop_rows - len(new_df)} rows with missing values")
        
        # Verify numeric columns
        numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        for col in numeric_cols:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
        logging.debug("Converted numeric columns to numeric type")
        
        # Predict
        logging.debug("Generating predictions")
        predictions = model.predict(new_df)
        print("Predictions:", predictions)
        
        # Save predictions
        os.makedirs('artifacts', exist_ok=True)
        predictions_path = os.path.join('artifacts', 'predictions.csv')
        pd.Series(predictions, name='income').to_csv(predictions_path, index=False)
        logging.debug(f"Predictions saved to {predictions_path}")
        
        return predictions
    
    except Exception as e:
        logging.error(f"Error in make_predictions: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_path> <new_data_path>")
        sys.exit(1)
    make_predictions(sys.argv[1], sys.argv[2])