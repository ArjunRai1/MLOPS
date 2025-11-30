import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
# test_data = pd.read_csv("./data\\processed\\test_preprocessed.csv")
def load_data(filepath : str) -> pd.DataFrame:
    try:
         return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}:{e}")
    

# X_test = test_data.drop(columns=['Potability'], axis=1)
# y_test = test_data['Potability']

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        X = data.drop(columns=['Potability'],axis=1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error Preparing data:{e}")
    
def load_model(filepath:str):
    try:
        with open(filepath,"rb") as file:
            model= pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}:{e}")

# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1score = f1_score(y_test, y_pred)

#     metrics_dict = {
#         "accuracy":accuracy,
#         "precision":precision,
#         "recall":recall,
#         "f1score":f1score
#     }

#     with open("metrics.json", "w") as file:
#         json.dump(metrics_dict, file, indent=4)

def evaluation_model(model, X_test:pd.DataFrame, y_test:pd.Series) -> dict:
    try:
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test,y_pred)
        pre = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1score = f1_score(y_test,y_pred)

        metrics_dict = {
            'acc':acc,
            'precision':pre,
            'recall' : recall,
            'f1_score': f1score
        }
        return metrics_dict
    
    except Exception as e:
        raise Exception(f"Error evaluating model : {e}")
    
def save_metrics(metrics:dict,metrics_path:str) -> None:
    try:
        with open(metrics_path,'w') as file:
            json.dump(metrics,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}:{e}")
    
def main():
    try:
        test_data_path = "./data/processed/test_preprocessed.csv"
        model_path = "model.pkl"
        metrics_path = "metrics.json"

        test_data = load_data(test_data_path)
        X_test,y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluation_model(model,X_test,y_test)
        save_metrics(metrics,metrics_path)
    except Exception as e:
        raise Exception(f"An Error occurred:{e}")

if __name__ == "__main__":
    main()