import mlflow.xgboost
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
import os
import mlflow
from mlflow import MlflowClient

def train_data():
    train_data= '/opt/ml/processing/input/train'
    
     # Read input files 
    input_files = [
        os.path.join(train_data, file) 
        for file in os.listdir(train_data) 
        if os.path.isfile(os.path.join(train_data, file))
    ]

    if not input_files:
        raise ValueError('No input files found in the training directory')
    
    raw_data = [pd.read_csv(file, header=None, engine="python") for file in input_files]
    data = pd.concat(raw_data)
    

    print(f"Training data shape =============: {data.shape}")

     # Set MLflow tracking URI (if needed)
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')


    if tracking_uri is None:
        print('uri not found in environment')
        tracking_uri = 'arn:aws:sagemaker:us-east-1:750573229682:mlflow-tracking-server/mlflow-tracking-server-sagemaker-poc'

    mlflow.set_tracking_uri(tracking_uri)

    X = data.drop(columns=['target'])  # Assuming 'target' is the name of the target column
    y = data['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Step 3: Log model in MLflow
    mlflow.set_experiment("xgboost_iris_experiment")
    with mlflow.start_run() as run:
        # Log model
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")
        print(f"Model logged in MLflow with run ID: {mlflow.active_run().info.run_id}")
        mlflow.log_param("test_size",0.2)
        mlflow.log_param("random_state",42)
        

if __name__ == "__main__":
    train_data()