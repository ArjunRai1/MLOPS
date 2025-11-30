import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

# data = pd.read_csv(r"C:\\Users\\arjun\\Downloads\\water_potability.csv")
def load_data(filepath: str)->pd.DataFrame:
    return pd.read_csv(filepath)

# test_size = yaml.safe_load(open("params.yaml", "r"))["data_collection"]["test_size"]
def load_params(filepath: str)->float:
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)["data_collection"]["test_size"]
    return params

# train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
def split(data: pd.DataFrame, test_size:float)->tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(data, test_size=test_size, random_state=42)

def save_csv(df:pd.DataFrame, filepath:str)->None:
    df.to_csv(filepath, index=False) 

#data_path = os.path.join("data", "raw")

def main():
    data_filepath = r"C:\\Users\\arjun\\Downloads\\water_potability.csv"
    params_filepath = "params.yaml"
    raw_data_path = os.path.join("data", "raw")
    data = load_data(data_filepath)
    test_size = load_params(params_filepath)
    train_data, test_data = split(data, test_size)
    os.makedirs(raw_data_path)

    save_csv(os.path.join(raw_data_path, "train.csv"))
    save_csv(os.path.join(raw_data_path, "test.csv"))

if __name__ == "main":
    main()