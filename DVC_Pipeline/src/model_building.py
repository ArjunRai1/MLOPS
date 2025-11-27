import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("./data/processed/train_preprocessed.csv")

X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=5, random_state=42)

rf.fit(X_train, y_train)

pickle.dump(rf, open("model.pkl", "wb"))