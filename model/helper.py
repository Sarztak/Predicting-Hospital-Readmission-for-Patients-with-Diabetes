from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from pathlib import Path
import pandas as pd 
import pickle

def split_data_by_patient_id(df, ratio="70:10:10:10"):
    """ratio is the ratio to split the data into train, validation, test, and calibration"""
    # get the unique patient ids
    patient_ids = df.patient_nbr.unique()

    # split by the ratio into train val, test, and calibration
    n_ratio = [
        int(len(patient_ids) * float(r) / 100) for r in ratio.split(":")
    ]
    train_ratio, val_ratio, test_ratio, calib_ratio = n_ratio

    X_train_ids, X_temp_ids = train_test_split(
        patient_ids, train_size=train_ratio, random_state=1984
    )

    X_val_ids, X_temp_ids = train_test_split(
        X_temp_ids, train_size=val_ratio / (val_ratio + test_ratio + calib_ratio), random_state=1984
    )

    X_test_ids, X_calib_ids = train_test_split(
        X_temp_ids, train_size=test_ratio / (test_ratio + calib_ratio), random_state=1984
    )

    def _make_X_y(ids):
        X_df = df.loc[df.patient_nbr.isin(ids)]
        y = X_df.readmitted
        X_df = X_df.drop(columns=['readmitted', 'patient_nbr'])
        return X_df, y

    # now once you have the split by patient ids then use the ids
    X_train, y_train = _make_X_y(X_train_ids)
    X_val, y_val = _make_X_y(X_val_ids)
    X_test, y_test = _make_X_y(X_test_ids)
    X_calib, y_calib = _make_X_y(X_calib_ids)

    split_data = dict(
        train=(X_train, y_train),
        val=(X_val, y_val),
        test=(X_test, y_test),
        calib=(X_calib, y_calib)
    )

    return split_data

if __name__ == "__main__":

    data_dir = Path("./data")
    df = pd.read_parquet(data_dir / "cleaned_data.parquet")
    split_data = split_data_by_patient_id(df)

    with open(data_dir / "split_data.pkl", 'wb') as f:
        pickle.dump(split_data, f)