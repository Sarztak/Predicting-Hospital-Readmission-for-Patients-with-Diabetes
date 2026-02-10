from rich.traceback import install; install()
import h2o; h2o.init();
from h2o.automl import H2OAutoML
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import (
    H2OGradientBoostingEstimator,
    H2OXGBoostEstimator,
    H2ORandomForestEstimator,
)
from pathlib import Path
from helper import split_data_by_patient_id
import pandas as pd 

if __name__ == "__main__":
    # load data
    data_dir = Path('./data')
    df = pd.read_parquet(data_dir / "cleaned_data.parquet")
    X_train, X_test, y_train, y_test = split_data_by_patient_id(df)