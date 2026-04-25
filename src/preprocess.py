## Preprocessing

# Import
import pandas as pd
import sys
import yaml
import os
from sklearn.model_selection import train_test_split


# Load config fron config.yaml
params = yaml.safe_load(open("params.yaml"))["preprocess"]

train_path = params["train_output"]
val_path = params["val_output"]
test_path = params["test_output"]

# Function to preprocess raw data
def preprocess():
    # Read raw data into pandas dataframe
    data = pd.read_csv(params["input"])

    # Split data into train, test and validation sets
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # Train-Test split
    train_x,X_test,train_y,y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    # Train-Validation spli
    X_train,X_val,y_train,y_val= train_test_split(train_x,train_y, test_size=0.10, random_state=44)
    
    
    # Create output directory
    os.makedirs("data/processed", exist_ok=True)
    
    # Save processed data to output path
    
    # Train CSV
    y_train = pd.Series(y_train, name="Outcome")
    train_df = pd.concat([X_train,y_train], axis=1)
    train_df.to_csv(params["train_output"], index=False)

    # Test CSV
    y_test = pd.Series(y_test, name="Outcome")
    test_df = pd.concat([X_test,y_test], axis=1)
    test_df.to_csv(params["test_output"], index=False)

    # Val CSV
    y_val = pd.Series(y_val, name = "Outcome")
    val_df = pd.concat([X_val,y_val], axis=1)
    val_df.to_csv(params["val_output"], index=False)



    print(f"Preprocessing Complete")

if __name__ == "__main__":
    # Call the preprocess function to preprocess raw data
    preprocess()