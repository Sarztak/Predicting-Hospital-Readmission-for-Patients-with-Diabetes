from rich.traceback import install; install()
from flaml import AutoML
from pathlib import Path
import pandas as pd 
import pickle
import mlflow
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, PrecisionRecallDisplay, precision_recall_curve
import matplotlib.pyplot as plt 
import numpy as np

if __name__ == "__main__":
    data_dir = Path('./data')

    with open(data_dir / "split_data.pkl", 'rb') as f:
        split_data = pickle.load(f)
    
    X_train, y_train = split_data["train"]
    X_val, y_val = split_data["val"]
    X_test, y_test = split_data["test"]

    # Compute class weights
    counts = y_train.value_counts().to_dict()
    total = sum(counts.values())
    n_classes = len(counts)
    class_weights = {_cls: total / (n_classes * n_count) for _cls, n_count in counts.items()}
    
    sample_weights = y_train.map(class_weights).values

    # Set MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("baseline_automl_flaml")

    with mlflow.start_run(run_name="flaml_search_no_class_weights"):
        
        automl = AutoML()
        
        # Log parent run parameters
        mlflow.log_param("time_budget", 60)
        mlflow.log_param("metric", "ap")
        mlflow.log_param("estimators", "xgboost,lgbm,rf")
        mlflow.log_param("seed", 1984)
        
        # Fit
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weight=sample_weights, 
            metric='ap', # average precision - same as AUCPR
            task='classification',
            estimator_list=['xgboost', 'lgbm', 'rf'],
            time_budget=60,
            seed=1984,
            verbose=3,
        )

        mlflow.log_param("best_estimator", automl.best_estimator)
        mlflow.log_dict(automl.best_config, "best_config.json")
        mlflow.log_metric("best_validation_loss", automl.best_loss)
        
        # Evaluate on validation set
        y_pred_proba_val = automl.predict_proba(X_val)[:, 1]
        val_aucpr = average_precision_score(y_val, y_pred_proba_val)
        val_auc = roc_auc_score(y_val, y_pred_proba_val)
        val_logloss = log_loss(y_val, y_pred_proba_val)
        
        mlflow.log_metric("val_aucpr", val_aucpr)
        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("val_logloss", val_logloss)
        
        # Evaluate on test set
        y_test_pred_proba = automl.predict_proba(X_test)[:, 1]
        test_aucpr = average_precision_score(y_test, y_test_pred_proba)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)
        test_logloss = log_loss(y_test, y_test_pred_proba)
        
        mlflow.log_metric("test_aucpr", test_aucpr)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_logloss", test_logloss)

        plt.figure(figsize=(10, 6))
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba_val) 
        PrecisionRecallDisplay.from_predictions(y_val, y_pred_proba_val)
        plt.savefig('./outputs/images/pr_curve.png')
        mlflow.log_artifact('./outputs/images/pr_curve.png')

        # precision and threshold at fixed recall 
        recall_fixed = 0.8
        op_idx = np.argmin(np.abs(recall - recall_fixed))
        precision_op = precision[op_idx]
        threshold_op = thresholds[op_idx]
        recall_op = recall[op_idx]

        # log the result as artifact
        mlflow.log_dict(
            dict(
                precision_op=precision_op,
                recall_op=recall_op,
                threshold_op=threshold_op,
            ),
            "model_operating_point_at_recall_0p8.json"
        )



        