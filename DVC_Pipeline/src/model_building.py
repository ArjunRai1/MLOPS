import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"], params["model_building"]["min_samples_leaf"], params["model_building"]["max_depth"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {params_path}: {e}")
    

# train_data = pd.read_csv("./data/processed/train_preprocessed.csv")
def load_data(data_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {e}")

# X_train = train_data.drop(columns=['Potability'], axis=1)
# y_train = train_data['Potability']
def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

# n_estimators = yaml.safe_load(open("params.yaml", "r"))["model_building"]["n_estimators"]
# min_samples_leaf = yaml.safe_load(open("params.yaml", "r"))["model_building"]["min_samples_leaf"]
# max_depth = yaml.safe_load(open("params.yaml", "r"))["model_building"]["max_depth"]

def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int, max_depth:int, min_samples_leaf:int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        clf.fit(X, y)
        return clf
    except Exception as e:
        raise Exception(f"Error training model: {e}")
    
# rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
# rf.fit(X_train, y_train)
def save_model(model: RandomForestClassifier, model_name: str) -> None:
    try:
        with open(model_name, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model to {model_name}: {e}")
# pickle.dump(rf, open("model.pkl", "wb"))

def main():
    try:
        params_path = "params.yaml"
        data_path = "./data/processed/train_preprocessed.csv"
        model_name = "model.pkl"

        n_estimators, min_samples_leaf, max_depth = load_params(params_path)
        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)

        model = train_model(X_train, y_train, n_estimators, max_depth, min_samples_leaf)
        save_model(model, model_name)
        print("Model trained and saved successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()