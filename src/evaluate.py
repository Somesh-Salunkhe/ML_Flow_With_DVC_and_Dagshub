## Evalution Script

# Imports
import pandas as pd 
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from dotenv import load_dotenv
import sys


# Load environment variables
load_dotenv()
for env_var in ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]:
    val = os.getenv(env_var)
    if val is not None:
        os.environ[env_var] = val

# Fix for Windows encoding issues with emojis in MLflow
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load configuration from params.yaml
params = yaml.safe_load(open("params.yaml"))['train']

def evaluate(data_path, model_path):
    # Read data
    data = pd.read_csv(data_path)

    # Separating labels
    X = data.drop(columns=["Outcome"])
    y = data['Outcome']

    # Set tracking URI
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    # Load model
    model = pickle.load(open(model_path, 'rb'))

    # Prediction
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Logging Metrics
    mlflow.log_metric("accuracy", accuracy)

    print(f"Model accuracy: {accuracy}")


if __name__ == "__main__":
    evaluate(params['test_data'], params['model'])