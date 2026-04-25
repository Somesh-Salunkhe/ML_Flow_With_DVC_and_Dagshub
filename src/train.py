## Trainer

# Imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse
from dotenv import load_dotenv
import mlflow
import sys

# Loading environment variables
load_dotenv()
for env_var in ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]:
    val = os.getenv(env_var)
    if val is not None:
        os.environ[env_var] = val

# Fix for Windows encoding issues with emojis in MLflow
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Function for hyperparamete tuning
def hyperparameter_tuning(X_train,y_train, param_grid):
    # Classifier
    rf = RandomForestClassifier()

    # Grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

# Load training config from config.yaml
params = yaml.safe_load(open("params.yaml"))["train"]


# Model trainer and ML flow logging function
def trainer(train_path, val_path, test_path , model_path):
    # Read processed data
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)

    # Splitting features and target
    X_train = train_data.drop(columns=["Outcome"])
    y_train = train_data["Outcome"]

    X_val = val_data.drop(columns=["Outcome"])
    y_val = val_data["Outcome"]

    X_test = test_data.drop(columns=["Outcome"])
    y_test = test_data["Outcome"]

    # Setting mlflow tracking uri
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    # Start ML Flow run
    with mlflow.start_run():
        
        # Signature for input and ouput schema
        signature = infer_signature(X_train, y_train)

        # Define hyperparameter grid
        param_grid = {
            'n_estimators' : [100,200],
            'max_depth' : [5,10,None],
            'min_samples_split' : [2,5],
            'min_samples_leaf' : [1,2]
        }

        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        # Best model
        best_model = grid_search.best_estimator_

        # Evaluate on validation set 
        val_pred = best_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)

        print(f"Validation Accuracy: {val_accuracy}")

        # Prediction
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {test_accuracy}")

        # Log metrics
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric('accuracy', test_accuracy)
        mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_samples_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_samples_leaf", grid_search.best_params_['min_samples_leaf'])

        # Log confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(str(cr), "classification_report.txt")

        # Verify tracking uri for logging of model
        tracking_uri = mlflow.get_tracking_uri()
        tracking_url_type_store = urlparse(tracking_uri).scheme

        if tracking_url_type_store not in ['file', ''] and not tracking_uri.startswith('mlruns'):
            mlflow.sklearn.log_model(best_model, "model", registered_model_name='Best model')
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        # Create directory to save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save model
        filename = model_path
        pickle.dump(best_model, open(filename, 'wb'))

        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    trainer( params["train_data"],
        params["val_data"],
        params["test_data"],
        params["model"])