import pandas as pd

def preprocess(df):
   # Define expected columns
       columns = [
           "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country"
       ]
       
       # Ensure input has correct columns
       df = df[columns]
       
       # Handle categorical features with one-hot encoding
       categorical_columns = ["workclass", "education", "marital-status", "occupation", 
                             "relationship", "race", "sex", "native-country"]
       df = pd.get_dummies(df, columns=categorical_columns)
       
       # Load training feature names (from log_all_models.py)
       try:
           train_data = pd.read_csv("data/adult.test", skiprows=1, header=None)
           train_data.columns = columns + ["income"]
           train_X = train_data.drop("income", axis=1)
           train_X = pd.get_dummies(train_X, columns=categorical_columns)
           train_features = train_X.columns.tolist()
       except FileNotFoundError:
           raise FileNotFoundError("data/adult.test required for feature alignment")
       
       # Align test features with training features
       missing_cols = set(train_features) - set(df.columns)
       for col in missing_cols:
           df[col] = 0
       df = df[train_features]
       
       return df