## Evalution Script

# Imports
import pandas as pd 
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from dotenv import load_dotenv


# Load environment variables
load_dotenv()
os.environ['MLFLOW_TRACKING_URI'] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Load configuration from config.yaml
params = yaml.safe_load(open("config.yaml"))['train']

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
    evaluate(params['data'], params['model'])