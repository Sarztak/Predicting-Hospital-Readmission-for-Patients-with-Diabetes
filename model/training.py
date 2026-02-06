import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from rich.traceback import install; install()
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn


def main(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(class_weight='balanced')
    
    # set the name of the experiment to run; an experiement is a container for the runs
    # if this is not set all runs go to a default experiment
    mlflow.set_experiment("rf_model")

    # Start MLflow experiment; this is the scope in which all the log_*() calls apply
    # you can also manually start and stop the runs but context manager is cleaner
    with mlflow.start_run(run_name="RandomForestClassifier"):
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
            roc_auc=roc_auc_score(y_test, y_pred_proba)
        )
        mlflow.log_metrics(metrics) 

        # log model
        mlflow.sklearn.log_model(rf, "random_forest_model")
        
        # log feature importance; log_dict logs to artifact
        feature_importance = dict(zip(X_train.columns, rf.feature_importances_))
        mlflow.log_dict(feature_importance, "feature_importance.json") # this is something new I learned

        # alternatively the feature importance can be logged as image
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance.values(), feature_importance.keys())
        plt.xlabel("Feature Importance")
        plt.title("Feature Importances")
        plt.savefig("./outputs/images/feature_importances.png")
        plt.close()
        mlflow.log_artifact("./outputs/images/feature_importances.png")

        # log confusion matrix as artifact; log_text logs to artifact
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_text(str(cm), "confusion_matrix.txt")


def split_data_by_patient_id(df):
    # get the unique patient ids
    patient_ids = df.patient_nbr.unique()

    # split by the patient_ids into train test split
    X_train_patient_ids, X_test_patient_ids, = train_test_split(patient_ids, test_size=0.2) 

    # now once you have the split by patient ids then use the ids to create train and test features 
    train_df = df.loc[df.patient_nbr.isin(X_train_patient_ids)]
    test_df = df.loc[df.patient_nbr.isin(X_test_patient_ids)]

    y_train = train_df.readmitted
    y_test = test_df.readmitted
    
    X_train = train_df.drop(columns=['readmitted', 'patient_nbr'])
    X_test = test_df.drop(columns=['readmitted', 'patient_nbr'])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # load data
    data_dir = Path('./data')
    df = pd.read_parquet(data_dir / "cleaned_data.parquet")
    X_train, X_test, y_train, y_test = split_data_by_patient_id(df)


    # set the tracking uri; if not set then everything will be written directly 
    # to disk in mlruns
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # run the experiment
    main(X_train, X_test, y_train, y_test)