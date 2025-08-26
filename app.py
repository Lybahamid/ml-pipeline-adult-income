from fastapi import FastAPI, File, UploadFile
import pandas as pd
import joblib
import preprocessing

app = FastAPI()

# Load the model
model = joblib.load("artifacts/best_model.joblib")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
       # Read CSV file
       df = pd.read_csv(file.file)
       
       # Preprocess data
       X = preprocessing.preprocess(df)
       
       # Make predictions
       predictions = model.predict(X).tolist()
       
       return {"predictions": predictions}