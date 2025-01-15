import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
import mlflow
from sklearn.datasets import load_digits  # Replace with your dataset

# Step 1: Load data (replace with your new dataset)
data = load_digits()  # Example using load_digits
X = pd.DataFrame(data.data, columns=[f"feature_{i}" for i in range(data.data.shape[1])])
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

tracking_uri = 'arn:aws:sagemaker:us-east-1:750573229682:mlflow-tracking-server/mlflow-tracking-server-sagemaker-poc'
mlflow.set_tracking_uri(tracking_uri) 

# Step 3: Log model in MLflow
mlflow.set_experiment("xgboost_digits_experiment")  # Change experiment name
with mlflow.start_run() as run:
    # Log model
    mlflow.xgboost.log_model(model, artifact_path="xgboost_model")
    print(f"Model logged in MLflow with run ID: {mlflow.active_run().info.run_id}")
    run_id = run.info.run_id

# Step 4: Load model from MLflow
model_uri = f"runs:/{run_id}/xgboost_model"
loaded_model = mlflow.xgboost.load_model(model_uri)

# Step 5: Predict on multiple samples one by one
samples = X_test.iloc[:5]  # Select first 5 samples from the test set

print("Predictions:")
for i, sample in samples.iterrows():
    sample = sample.values.reshape(1, -1)
    prediction = loaded_model.predict(sample)
    print(f"Sample {i}: Predicted class = {prediction[0]}")

# Evaluate model accuracy
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy}")
