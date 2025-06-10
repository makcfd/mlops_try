if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@custom
def train_model(*args, **kwargs):
    mlflow.sklearn.autolog()
    df_train, df_val = train_test_split(df, test_size=df_split, random_state=random_state)
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_train = df_train[target].values
    y_val = df_val[target].values
    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)

        loss = root_mean_squared_error(y_val, y_pred)

    print(f"rmse: {loss}")
    print(f"Model intercept: {lr.intercept_}")
    return dv, lr

    return {}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
