from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import joblib
import pandas as pd
from preprocessing import get_preprocessing_pipeline
import os
import numpy as np
import logging  # Added import

np.random.seed(42)

def train_models(df):
    """
    Train models, tune hyperparameters, save best model.
    """
    logging.debug("Starting train_models")
    try:
        # Split data
        X = df.drop('income', axis=1)
        y = df['income']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Get preprocessing pipeline
        preprocess_pipeline = get_preprocessing_pipeline()
        
        # Models and params
        models = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {'classifier__C': [0.1, 1, 10]}
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10, 20]}
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.01, 0.1]}
            }
        }
        
        best_model = None
        best_score = 0
        best_name = ''
        
        for name, info in models.items():
            # Full pipeline: preprocess + model
            pipeline = Pipeline(steps=[
                ('preprocessing', preprocess_pipeline),
                ('classifier', info['model'])
            ])
            
            # Grid search
            logging.debug(f"Training {name}")
            grid = GridSearchCV(pipeline, info['params'], cv=5, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            # Evaluate on test
            y_pred = grid.predict(X_test)
            score = f1_score(y_test, y_pred)
            print(f"{name} Best F1: {score:.3f}")
            logging.debug(f"{name} Best F1: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_model = grid.best_estimator_
                best_name = name
        
        # Save best model
        os.makedirs('artifacts', exist_ok=True)
        model_path = os.path.join('artifacts', f'best_model_{best_name}.joblib')
        joblib.dump(best_model, model_path)
        logging.debug(f"Best model ({best_name}) saved to {model_path}")
        
        return best_model, X_test, y_test
    
    except Exception as e:
        logging.error(f"Error in train_models: {e}")
        raise

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    df = pd.read_csv(os.path.join('artifacts', 'cleaned_data.csv'))
    train_models(df)