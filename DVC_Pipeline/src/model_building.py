import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

train_data = pd.read_csv("./data/processed/train_preprocessed.csv")

X_train = train_data.drop(columns=['Potability'], axis=1)
y_train = train_data['Potability']

n_estimators = yaml.safe_load(open("params.yaml", "r"))["model_building"]["n_estimators"]
min_samples_leaf = yaml.safe_load(open("params.yaml", "r"))["model_building"]["min_samples_leaf"]
max_depth = yaml.safe_load(open("params.yaml", "r"))["model_building"]["max_depth"]

rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)

rf.fit(X_train, y_train)

pickle.dump(rf, open("model.pkl", "wb"))