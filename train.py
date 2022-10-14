import mlfoundry
from datetime import datetime
from train_model import train_model
from dataset import get_initial_data

train_data_path = "./kaggle_twitter/Data/train.csv"
test_data_path = "./kaggle_twitter/Data/test.csv"

X_train, X_test, y_train, y_test = get_initial_data(train_data_path, test_data_path)

model, metadata = train_model(X_train, y_train, X_test, y_test)


run = mlfoundry.get_client().create_run(project_name="twitter-sentiment", run_name=f"train-{datetime.now().strftime('%m-%d-%Y')}")
run.log_params(model.get_params())

run.log_metrics({
    'train/accuracy_score': metadata["accuracy_train"][-1],
    'train/f1': f1_scor,
    'test/accuracy_score': metadata["accuracy_test"][-1],
    'test/f1': f1_scor})

run.log_dataset(
    dataset_name='train',
    features=x_train_bow,
    actuals=y_train_bow,
)

run.log_dataset(
    dataset_name='test',
    features=x_test_bow,
    predictions=y_pred,
    actuals=y_test_bow,
)

model_version = run.log_model(
    name="KNN-classifier",
    model=model,
    framework="sklearn",
    description="model trained for red wine quality classification",
    metadata=metadata,
    model_schema=schema,
    custom_metrics=[{"name": "f1_score", "type": "metric", "value_type": "float"}],
)

print(f"Logged model: {model_version.fqn}")