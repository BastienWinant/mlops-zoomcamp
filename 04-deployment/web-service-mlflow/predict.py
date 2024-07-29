import pickle
from flask import Flask, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient

RUN_ID = 'b4d...'
TRACKING_URI="http:127.0.0.1:5000"
EXPERIMENT_NAME="green-taxi-duration"

# mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
mlflow.set_tracking_uri(uri=TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)

local_path = client.download_artifacts(run_id=RUN_ID, path='dict_vectorizer.bin')
print(f"Downloading preprocessor to {local_path}")
with open(local_path, 'rb') as f_in:
  dv = pickle.load(f_in)
  

with open("lin_reg.bin", "rb") as f_in:
  (dv, model) = pickle.load(file=f_in)


def prepare_features(ride):
  features = {
    "PU_DO": f'{ride["PULocationID"]}_{ride["DOLocationID"]}',
    "trip_distance": ride["trip_distance"]
  }

  return features

def predict(features):
  X = dv.transform(features)
  preds = model.predict(X)

  return preds[0]


app = Flask('duration-prediction')

@app.route("/predict", methods=['POST'])
def predict_endpoint():
  ride = request.get_json()
  features = prepare_features(ride)
  pred = predict(features)

  result = {
    "duration": pred,
    "model_version": RUN_ID
  }

  return jsonify(result)

if __name__=="__main__":
  app.run(debug=True, host="0.0.0.0", port=9696)