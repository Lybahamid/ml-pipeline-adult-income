import logging
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set matplotlib backend for non-interactive plotting
plt.switch_backend('agg')

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model, save metrics and plots to artifacts/.
    """
    logging.debug("Starting evaluate_model")
    try:
        # Generate predictions
        logging.debug("Generating predictions")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        logging.debug("Calculating metrics")
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None
        }
        
        # Save metrics
        artifacts_dir = 'artifacts'
        logging.debug(f"Ensuring artifacts directory exists: {artifacts_dir}")
        os.makedirs(artifacts_dir, exist_ok=True)
        metrics_path = os.path.join(artifacts_dir, 'metrics.json')
        logging.debug(f"Saving metrics to {metrics_path}")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.debug(f"Metrics saved: {metrics}")
        
        # Confusion matrix plot
        logging.debug("Generating confusion matrix plot")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(artifacts_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        logging.debug(f"Confusion matrix saved to {cm_path}")
        
        # ROC curve plot
        if y_prob is not None:
            logging.debug("Generating ROC curve plot")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='best')
            roc_path = os.path.join(artifacts_dir, 'roc_curve.png')
            plt.savefig(roc_path, bbox_inches='tight')
            plt.close()
            logging.debug(f"ROC curve saved to {roc_path}")
        else:
            logging.warning("No predict_proba available, skipping ROC curve")
        
        print("Evaluation complete. Metrics:", metrics)
        return metrics
    
    except Exception as e:
        logging.error(f"Error in evaluate_model: {e}")
        raise

if __name__ == "__main__":
    try:
        import joblib
        import pandas as pd
        from train import train_models
        logging.debug("Loading cleaned data")
        df = pd.read_csv(os.path.join('artifacts', 'cleaned_data.csv'))
        logging.debug("Running train_models")
        model, X_test, y_test = train_models(df)
        logging.debug("Running evaluate_model")
        evaluate_model(model, X_test, y_test)
    except Exception as e:
        logging.error(f"Error in main block: {e}")