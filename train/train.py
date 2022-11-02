import mlfoundry
from datetime import datetime
from train_model import train_model
from dataset import get_initial_data
import pickle

train_data_path = "train.csv"
test_data_path = "test.csv"

X_train, X_test, y_train, y_test, bow = get_initial_data(train_data_path, test_data_path)

model, metadata = train_model(X_train, y_train, X_test, y_test)


run = mlfoundry.get_client().create_run(project_name="twitter-sentiment", run_name=f"train-{datetime.now().strftime('%m-%d-%Y')}")
run.log_params(model.get_params())


with open('vectorizer.pickle', 'wb') as fout:
    pickle.dump((bow), fout)
    
run.log_metrics({
    'train/accuracy_score': metadata["accuracy_train"][-1],
    'train/f1': metadata["f1_scor_train"],
    'test/accuracy_score': metadata["accuracy_test"][-1],
    'test/f1': metadata["f1_scor_test"]})

run.log_dataset(
    dataset_name='train',
    features=X_train,
    actuals=y_train)

run.log_dataset(
    dataset_name='test',
    features=X_test,
    predictions=metadata["y_pred"],
    actuals=y_test,
)

model_version = run.log_model(
    name="KNN-classifier",
    model=model,
    framework="sklearn",
    description="model trained for twitter sentiment analysis",
)
model_artifact = run.log_artifact(local_path="vectorizer.pickle", artifact_path="my-artifacts")

print(f"Logged model: {model_version.fqn}")
