from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from pathlib import Path
import pandas as pd 
import pickle

def split_data_by_patient_id(df, ratio="70:10:10:10"):
    """ratio is the ratio to split the data into train, validation, test, and calibration"""
    # get the unique patient ids
    patient_ids = df.patient_nbr.unique()
    patient_ids_shuffled = shuffle(patient_ids, random_state=1984)

    # split by the ratio into train val, test, and calibration
    n_ratio = [
        int(len(patient_ids) * float(r) / 100) for r in ratio.split(":")
    ]

    _t = 0 
    n_len = []
    for n in n_ratio:
        _t += n
        n_len.append(_t) # this is to find the endpoints of the splits

    n_train, n_val, n_test, n_calib = n_len
    # split the patients_ids into train, val, test, calib
    X_train_patient_ids = patient_ids_shuffled[:n_train]
    X_val_patient_ids = patient_ids_shuffled[n_train:n_val]
    X_test_patient_ids = patient_ids_shuffled[n_val:n_test]
    X_calib_patient_ids = patient_ids_shuffled[n_test:]

    def _make_X_y(ids):
        X_df = df.loc[df.patient_nbr.isin(ids)]
        y = X_df.readmitted
        X_df = X_df.drop(columns=['readmitted', 'patient_nbr'])
        return X_df, y

    # now once you have the split by patient ids then use the ids
    X_train, y_train = _make_X_y(X_train_patient_ids)
    X_val, y_val = _make_X_y(X_val_patient_ids)
    X_test, y_test = _make_X_y(X_test_patient_ids)
    X_calib, y_calib = _make_X_y(X_calib_patient_ids)

    split_data = dict(
        train=(X_train, y_train),
        val=(X_val, y_val),
        test=(X_test, y_test),
        calib=(X_calib, y_calib)
    )

    return split_data
def log_h2o_runs():
    import mlflow
    import h2o
    from h2o.automl import H2OAutoML

    # Start MLflow run
    with mlflow.start_run(run_name="h2o_automl_experiment"):
        
        # Log parameters
        mlflow.log_param("max_models", 50)
        mlflow.log_param("max_runtime_secs", 300)
        mlflow.log_param("stopping_metric", "AUCPR")
        mlflow.log_param("algorithms", "XGBoost,GBM,DRF")
        
        # Train AutoML
        aml = H2OAutoML(
            max_models=50,
            seed=1984,
            nfolds=0,
            stopping_metric="AUCPR",
            sort_metric="AUCPR",
            include_algos=["XGBoost", "GBM", "DRF"],
            balance_classes=False,
            max_runtime_secs=300,
        )
        
        aml.train(x=x_cols, y=y_col, training_frame=train_h2o, validation_frame=val_h2o)
        
        # Get best model
        best_model = aml.leader
        
        # Log metrics
        perf = best_model.model_performance(val_h2o)
        mlflow.log_metric("aucpr", perf.aucpr())
        mlflow.log_metric("auc", perf.auc())
        mlflow.log_metric("logloss", perf.logloss())
        
        # Log model type
        mlflow.log_param("best_model_type", best_model.algo)
        mlflow.log_param("best_model_id", best_model.model_id)
        
        # Save and log model
        model_path = h2o.save_model(best_model, path="./h2o_model")
        mlflow.log_artifact(model_path)
        
        # Optional: Log leaderboard
        leaderboard = aml.leaderboard.as_data_frame()
        leaderboard.to_csv("leaderboard.csv", index=False)
        mlflow.log_artifact("leaderboard.csv")

if __name__ == "__main__":

    data_dir = Path("./data")
    df = pd.read_parquet(data_dir / "cleaned_data.parquet")
    split_data = split_data_by_patient_id(df)

    with open(data_dir / "split_data.pkl", 'wb') as f:
        pickle.dump(split_data, f)