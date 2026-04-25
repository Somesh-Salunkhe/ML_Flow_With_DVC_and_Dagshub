## Preprocessing

# Import
import pandas as pd
import sys
import yaml
import os
from sklearn.model_selection import train_test_split


# Load config fron config.yaml
params = yaml.safe_load(open("config.yaml"))["preprocess"]

# Function to preprocess raw data
def preprocess(input_path, output_paths):
    # Read raw data into pandas dataframe
    data = pd.read_csv(input_path)

    # Split data into train, test and validation sets
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    # Train-Test split
    train_x,X_test,train_y,y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    # Train-Validation spli
    X_train,X_val,y_train,y_val= train_test_split(train_x,train_y, test_size=0.10, random_state=44)
    
    
    # Create output directory
    for path in output_paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save processed data to output path
    
    # Train CSV
    y_train = pd.Series(y_train, name="Outcome")
    train_df = pd.concat([X_train,y_train], axis=1)
    train_df.to_csv(output_paths['train'], index=False)

    # Test CSV
    y_test = pd.Series(y_test, name="Outcome")
    test_df = pd.concat([X_test,y_test], axis=1)
    test_df.to_csv(output_paths['test'], index=False)

    # Val CSV
    y_val = pd.Series(y_val, name = "Outcome")
    val_df = pd.concat([X_val,y_val], axis=1)
    val_df.to_csv(output_paths['val'], index=False)



    print(f"Preprocessed data saved to {output_paths}")

if __name__ == "__main__":
    # Call the preprocess function to preprocess raw data
    preprocess(params['input'], params['output'])