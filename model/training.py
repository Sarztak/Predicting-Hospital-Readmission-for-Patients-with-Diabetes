import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score
from rich.traceback import install; install()
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import pickle

def main(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(class_weight="balanced")
    
    # set the name of the experiment to run; an experiement is a container for the runs
    # if this is not set all runs go to a default experiment
    mlflow.set_experiment("rf_model")

    # Start MLflow experiment; this is the scope in which all the log_*() calls apply
    # you can also manually start and stop the runs but context manager is cleaner
    with mlflow.start_run(run_name="rf_model_class_weights"):
        # train the model
        rf.fit(X_train, y_train)

        # make prediction
        y_pred_proba = rf.predict_proba(X_test)[:, 1]

        # log_param, log_metric, log_artifact and log_model
        params = dict(
            n_estimator=rf.n_estimators,
            max_depth=rf.max_depth,
            class_weight="balanced"
        )
        mlflow.log_params(params)

        # log metrics
        y_pred = (y_pred_proba > 0.56) # just a random choosen threshold
        metrics = dict(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
            roc_auc=roc_auc_score(y_test, y_pred_proba),
            aucpr=average_precision_score(y_test, y_pred_proba),
        )

        mlflow.log_metrics(metrics) 

        # log model
        mlflow.sklearn.log_model(rf, "random_forest_model")
        
        # log feature importance; log_dict logs to artifact
        feature_importance = tuple(zip(X_train.columns, rf.feature_importances_))
        feature_importance = sorted(feature_importance, key=lambda x: x[1])

        # select the top 5 features
        top_5_features = dict(feature_importance[-5:])
        mlflow.log_dict(top_5_features, "feature_importance.json") # this is something new I learned

        # alternatively the feature importance can be logged as image
        plt.figure(figsize=(15, 6))
        plt.barh(top_5_features.keys(), top_5_features.values())
        plt.xlabel("Feature Importance")
        plt.title("Feature Importances Top 5")
        plt.savefig("./outputs/images/feature_importances.png")
        plt.close()
        mlflow.log_artifact("./outputs/images/feature_importances.png")

        # log confusion matrix as artifact; log_text logs to artifact
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_text(str(cm), "confusion_matrix.txt")


if __name__ == "__main__":
    data_dir = Path('./data')

    with open(data_dir / "split_data.pkl", 'rb') as f:
        split_data = pickle.load(f)
    
    X_train, y_train = split_data["train"]
    X_val, y_val = split_data["val"]
    X_test, y_test = split_data["test"]

    # set the tracking uri; if not set then everything will be written directly 
    # to disk in mlruns
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # run the experiment
    main(X_train, X_test, y_train, y_test)
    
    # command to run the server which needs to be started before anything else
    # mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000