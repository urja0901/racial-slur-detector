import mlfoundry as mlf
import pandas as pd
import yaml

with open("infer.yaml", "r") as stream:
    try:
        env_vars = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Load the model from MLFoundry by proving the MODEL_FQN
client = mlf.get_client(api_key=env_vars['components'][0]['env']['MLF_API_KEY'],tracking_uri=env_vars['components'][0]['env']['MLF_HOST'])
model_version = client.get_model(env_vars['components'][0]['env']['MODEL_FQN'])
model = model_version.load()

def infer_model(sentiment_text):
    to_predict = [sentiment_text]

    test = pd.DataFrame(data=[to_predict],columns=['sentiment_text'])
    test = test.astype({'sentiment_text':'str'})
    prediction = model.predict(test)
    return {'sentiment':prediction}