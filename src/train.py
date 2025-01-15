import argparse
import os
import joblib
import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train(train_data, max_depth=6):
    """
    Train an XGBoost classifier.

    Args:
        train_data (pd.DataFrame): Training data with last column as target.
        max_depth (int, optional): Maximum depth of the tree. Defaults to 6.

    Returns:
        xgboost.XGBClassifier: Trained model.
    """
    # Separate features and target
    # Step 2: Split the data into features (X) and target (y)
    print(train_data)
    
    X = train_data.drop(columns=['target'])  # Assuming 'target' is the name of the target column
    y = train_data['target']

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    return model

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/processing/output')) 
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/processing/input/train')) 

    args = parser.parse_args()

    # Read input files 
    input_files = [
        os.path.join(args.train, file) 
        for file in os.listdir(args.train) 
        if os.path.isfile(os.path.join(args.train, file))
    ]

    print(f"Input files: {input_files}") 

    if not input_files:
        raise ValueError('No input files found in the training directory')

    # Read and concatenate data
    raw_data = [pd.read_csv(file, header=None, engine="python") for file in input_files]
    train_data = pd.concat(raw_data)

    print(f"Training data shape: {train_data.shape}")

    # Set MLflow tracking URI
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'arn:aws:sagemaker:us-east-1:750573229682:mlflow-tracking-server/mlflow-tracking-server-sagemaker-poc')
    mlflow.set_tracking_uri(tracking_uri) 

    # Enable MLflow autologging
    mlflow.xgboost.autolog() 
    print("Autologging enabled") 

    # Train the model
    with mlflow.start_run():
        model = train(train_data, args.max_depth)

        # Save the model
        model_path = os.path.join(args.model_dir, "model.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved at {model_path}")

        # Register the model with MLflow
        artifact_path = "iris-xgboost_model"
        mlflow.xgboost.log_model(model, artifact_path=artifact_path)

        print(f"Model logged to MLflow under artifact path: {artifact_path}")

if __name__ == '__main__':
    main()
