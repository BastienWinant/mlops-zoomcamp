# Model Deployment
In the design stage, we decide whether ML is the right solution
During training, we finetune a model and capture the relevant parameters/metrics
The output of training is a model that must be deployed
## Deployment Options
### Batch Deployment
- when predictions are not needed immediately
- runs on regular intervals
- pulls data from the previous period from the DB and writes the prediction
- often used in marketing related tasks (predicting churn)
### Online Deployment
#### Web Service
- receives HTTP requests and replies with prediction
- common way of deploying models upon user web interactions in the web
- up and running at all times
#### Streaming
- listens for events and respons with prediction
- producers (back-end) send events to consumers (services)
- the producer who originates the trigger is not the one waiting for the response
- example case: content moderation => youtube video uploads must be checked against violence, copyrights, etc.