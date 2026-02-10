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
import pickle

if __name__ == "__main__":
    # load data
    data_dir = Path('./data')

    with open(data_dir / "split_data.pkl", 'rb') as f:
        split_data = pickle.load(f)
    
    X_train, y_train = split_data["train"]
    X_val, y_val = split_data["val"]
    X_test, y_test = split_data["test"]

    # apply weights only to the training dataset
    # there are some preprocessing quirks that needs to be done in order to work with H2O
    # 1. the dataframe must include the target as well 
    # 2. the training dataframe should have a column called weights
    # 3. Pandas dataframe needs to be converted to H2O dataframe using h2o.H2OFrame(df)
    # 4. the target column needs to be converted to a factor because h2o is dumb and it will treat 0 and 1 as numerical and run a regression instead of classification. (Also it does not matter if you have already converted to categorical in pandas dataframe, it won't work)
    # 5. Despite the fact that features and target should be in the same dataframe still name of the feature columns needs to be provided separately

    X_train["target"] = y_train
    X_val["target"] = y_val 
    X_test["target"] = y_test

    counts = X_train["target"].value_counts().to_dict()
    total = sum(counts.values())
    n_classes = len(counts)
    class_weights = {_cls: total / (n_classes * n_count) for _cls, n_count in counts.items()}

    X_train["weights"] = X_train["target"].map(class_weights)
    
    X_train_h2o = h2o.H2OFrame(X_train)
    X_val_h2o = h2o.H2OFrame(X_val)
    X_test_h2o = h2o.H2OFrame(X_test)

    X_train_h2o["target"] = X_train_h2o["target"].asfactor()
    X_val_h2o["target"] = X_val_h2o["target"].asfactor()
    X_test_h2o["target"] = X_test_h2o["target"].asfactor()

    target_col_name = "target"
    feature_cols_name = [col for col in X_train.columns if col not in ["target", "weights"]]

    # after all this drama we can initialize aml object;
    # again there are certain things to keep in mind 
    # cross validation needs to be disabled because validation is handled externally
    # class weights needs to be disabled because they too are handled externally 
    # why use AUCPR ? Because it is the average precision across all the recall levels
    # When threshold is lowered the recall increases but precision tanks, so AUCPR answers the question that on average how much precision you have if you vary the recall. 
    # the better AUCPR the more precision on average even if you vary the recall 
    # High AUCPR (close to 1.0): You maintain high precision even as you increase recall. The model confidently separates classes.
    # Low AUCPR (close to 0): To increase recall even slightly, precision tanks. The model struggles.
    # AUCPR = what is the expected precision level if I pick up a random recall level
    # only XGBoost, GBM, and DRF (Distributed Random Forest, h2o implimentation of RF) these three because later the parameters needs to be ported to sklearn or xgboost models since h2o is tedious to use in production and you don't want to be tied to h2o

    aml = H2OAutoML(
        max_models=50,
        seed=1984,
        nfolds=0, # disable internal cv
        stopping_metric="AUCPR",
        sort_metric="AUCPR",
        include_algos=["XGBoost", "GBM", "DRF"],
        balance_classes=False,
        max_runtime_secs=300,
    )

    # now training 
    aml.train(
        x=feature_cols_name,
        y=target_col_name,
        training_frame=X_train_h2o,
        validation_frame=X_val_h2o,
        weights_column="weights"
    )

    leader = aml.leader
    test_perf = leader.model_performance(X_test_h2o)
    print("Test AUCPR:", test_perf.aucpr())