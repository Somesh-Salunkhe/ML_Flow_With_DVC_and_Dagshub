## Preprocessing

# Import
import pandas as pd
import sys
import yaml
import os


# Load config fron config.yaml
params = yaml.safe_load(open("config.yaml"))["preprocess"]

# Function to preprocess raw data
def preprocess(input_path, output_path):
    # Read raw data into pandas dataframe
    data = pd.read_csv(input_path)

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save processed data to output path
    data.to_csv(output_path, header=None, index=False)

    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    # Call the preprocess function to preprocess raw data
    preprocess(params['input'], params['output'])