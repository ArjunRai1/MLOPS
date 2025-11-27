import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
test_data = pd.read_csv("./data\\processed\\test_preprocessed.csv")

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)

    metrics_dict = {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1score":f1score
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics_dict, file, indent=4)