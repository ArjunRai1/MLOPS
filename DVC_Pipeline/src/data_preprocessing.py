import numpy as np
import pandas as pd
import os

# train_data = pd.read_csv("./data\\raw\\train.csv")
# test_data = pd.read_csv("./data\\raw\\test.csv")
def load_data(filepath: str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        return Exception(f"Failed to load data from {filepath}: {e}")

def simple_preprocess(df):
    try:
        for column in df.columns:
            median = df[column].median()
            df[column].fillna(median, inplace=True)
        return df
    except Exception as e:
        return Exception(f"Failed to perform pre-processing: {e}")

# train_preprocess = simple_preprocess(train_data)
# test_preprocess = simple_preprocess(test_data)
def save_data(df:pd.DataFrame, filepath:str)->None:
    try:
        df.to_csv(filepath, index=False) 
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")

def main():
    try:
        raw_data_path = "./data/raw"
        processed_data_path = "./data/processed"

        train_data = load_data(os.path.join(raw_data_path, "train.csv"))
        test_data = load_data(os.path.join(raw_data_path, "test.csv"))

        train_processed_data = simple_preprocess(train_data)
        test_processed_data = simple_preprocess(test_data)

        os.makedirs(processed_data_path)

        save_data(train_processed_data, os.path.join(processed_data_path, "train_preprocessed.csv"))
        save_data(test_processed_data, os.path.join(processed_data_path, "test_preprocessed.csv"))
    except Exception as e:
        raise Exception(f"Error occurred: {e}")
if __name__ == "__main__":
    main()