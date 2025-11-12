# src/train.py
"""
Train a Decision Tree classifier using CSV data, log metrics and model to MLflow.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import argparse
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


def main(max_depth: int, random_state: int, version: str, stratify: str = "NO"):
    # --------------------------
    # Setup MLflow
    # --------------------------
    mlflow.set_tracking_uri("http://127.0.0.1:8100")
    mlflow.set_experiment("Iris_DT_Classification_Exp")

    with mlflow.start_run():
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("iris_data_version", version)
        mlflow.log_param("stratify", stratify)

        # --------------------------
        # Load Local Data
        # --------------------------
        data_path = "data/iris.csv"
        print(f"Loading local data from {data_path}...")
        data = pd.read_csv(data_path)

        # --------------------------
        # Prepare Features and Target
        # --------------------------
        X = data.drop("species", axis=1)
        y = data["species"]

        # --------------------------
        # Train/Test Split
        # --------------------------
        if stratify.lower() == "yes":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )

        # --------------------------
        # Train Model
        # --------------------------
        print("Training Decision Tree model...")
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy_score = metrics.accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy_score)
        print(f"Accuracy: {accuracy_score:.3f}")
        
        mlflow.set_tag("Training Info","Decision Tree Model for IRIS data")

        # --------------------------
        # Save Model
        # --------------------------
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/model.joblib"
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # Log model to MLflow
        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="IRIS-Classifier-dt",
        )

        print("Training and logging complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Decision Tree on Iris dataset with MLflow logging")
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--version", type=str, required=True, help="Feature view version, e.g., v1")
    parser.add_argument("--stratify", type=str, default="NO")
    args = parser.parse_args()

    main(args.max_depth, args.random_state, args.version, args.stratify)
