import os
import pickle
import click
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import mlflow
from sklearn.feature_selection import VarianceThreshold

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("random-forest-train")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="../data/output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    mlflow.sklearn.autolog()
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_train = X_train.astype(np.float32)
    X_val   = X_val.astype(np.float32)

    # drop any feature with variance < 0.01
    vt = VarianceThreshold(threshold=0.01)
    X_train = vt.fit_transform(X_train)
    X_val   = vt.transform(X_val)
    with mlflow.start_run():
        rf = RandomForestRegressor(
            n_jobs=1,
            max_depth=6,
            random_state=0,
            max_leaf_nodes=20,
            n_estimators=10,
            min_samples_leaf=20,
            bootstrap=True,
            max_features="sqrt", 
            max_samples=0.6, 
            )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        print(rmse)

if __name__ == '__main__':
    run_train()
