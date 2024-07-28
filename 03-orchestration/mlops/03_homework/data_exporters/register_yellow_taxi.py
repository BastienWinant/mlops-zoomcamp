import pickle
import mlflow

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

mlflow.set_tracking_uri('http://mlflow:5000')
mlflow.set_experiment("nyc-taxi-regressor")

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    with open("./preprocessor.b", "wb") as f_out:
        pickle.dump(data['preprocessor'], f_out)

    with mlflow.start_run():
        mlflow.sklearn.log_model(sk_model=data["model"], artifact_path="model")
        mlflow.log_artifact("./preprocessor.b", artifact_path="preprocessor")



