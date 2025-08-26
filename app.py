from fastapi import FastAPI, UploadFile
import joblib
import pandas as pd
import io
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()
model = joblib.load(os.path.join('artifacts', 'best_model_XGBoost.joblib'))

@app.post("/predict")
async def predict(file: UploadFile):
    logging.debug("Received file upload for prediction")
    try:
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country'
        ]
        # Read and decode file content
        content = await file.read()  # Await the read operation to get bytes
        content_str = content.decode('utf-8')  # Decode bytes to string
        # Skip first row and select only 14 feature columns
        df = pd.read_csv(io.StringIO(content_str), skiprows=1, names=columns + ['income'], skipinitialspace=True)
        df = df.drop(['fnlwgt', 'income'], axis=1)  # Drop fnlwgt and income
        df = df.replace('?', pd.NA).dropna()
        df[['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']] = df[['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']].apply(pd.to_numeric, errors='coerce')
        logging.debug("Preprocessed uploaded data")
        preds = model.predict(df)
        logging.debug("Generated predictions")
        return {"predictions": preds.tolist()}
    except Exception as e:
        logging.error(f"Error in predict endpoint: {e}")
        return {"error": str(e)}