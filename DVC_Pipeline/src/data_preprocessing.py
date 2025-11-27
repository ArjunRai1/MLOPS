import numpy as np
import pandas as pd
import os

train_data = pd.read_csv("./data\\raw\\train.csv")
test_data = pd.read_csv("./data\\raw\\test.csv")

def simple_preprocess(df):
    for column in df.columns:
        median = df[column].median()
        df[column].fillna(median, inplace=True)
    return df

train_preprocess = simple_preprocess(train_data)
test_preprocess = simple_preprocess(test_data)

data_path = os.path.join("data", "processed")

os.makedirs(data_path)

train_preprocess.to_csv(os.path.join(data_path, "train_preprocessed.csv"), index=False)
test_preprocess.to_csv(os.path.join(data_path, "test_preprocessed.csv"), index=False)