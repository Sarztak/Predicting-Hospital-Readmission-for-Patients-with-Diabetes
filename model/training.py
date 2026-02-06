from sklearn.model_selection import train_test_split
from rich.traceback import install; install()
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn


def main(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(class_weight='balanced')
    
    # set the name of the experiment to run, this is the main
    mlflow.set_experiment("rf_model")

    # Start MLflow experiment
    with mlflow.start_run(run_name="RandomForestClassifier"):
        # train the model
        rf.fit(X_train, y_train)

        # make prediction
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
       

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
    X_test = train_df.drop(columns=['readmitted', 'patient_nbr'])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # load data
    data_dir = Path('./data')
    df = pd.read_parquet(data_dir / "cleaned_data.parquet")
    X_train, X_test, y_train, y_test = split_data_by_patient_id(df)
    main(X_train, X_test, y_train, y_test)