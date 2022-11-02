from fastapi import FastAPI
from pydantic import BaseModel
import mlfoundry as mlf
import pandas as pd
import yaml
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pickle

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = FastAPI(docs_url="/")


@app.get("/")
async def root():
    return {"message": "Welcome to Twitter sentiment analysis inference"}

class SentimentAnalysis(BaseModel):
    tweet: str
 
with open("infer.yaml", "r") as stream:
    try:
        env_vars = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Load the model from MLFoundry by proving the MODEL_FQN
client = mlf.get_client(api_key=env_vars['components'][0]['env']['MLF_API_KEY'],tracking_uri=env_vars['components'][0]['env']['MLF_HOST'])
model_version = client.get_model(env_vars['components'][0]['env']['MODEL_FQN'])
model = model_version.load()

def preprocessed_tweet(test):
    text = test['tweet'].values.tolist()

    stopword=nltk.corpus.stopwords.words('english')
    stopword.remove('not')
    for index,text_ in enumerate(text):
        text_=re.sub(r'@[\w]*','',text_) #Removing Twitter Handles (@user)
        text_=re.sub(r'http/S+','',text_) #Removing urls from text 
        text_=re.sub(r'[^A-Za-z#]',' ',text_) #Removing Punctuations, Numbers, and Special Characters
        text_=" ".join(i.lower() for i in text_.split() if i.lower() not in stopword) #Removing stopword
        text[index]=text_

    #Stemming the word
    pt=PorterStemmer()
    wordnet=WordNetLemmatizer()
    for index,text_ in enumerate(text):
        text_=" ".join(pt.stem(i) for i in text_.split())
        text_=" ".join(wordnet.lemmatize(i) for i in text_.split())  
        text[index]=text_

    df_test = pd.DataFrame()
    df_test['preprocess_tweet'] = text

    run = client.get_run(env_vars['components'][0]['env']['RUN_ID'])

    vectorizer_path = run.download_artifact(path="my-artifacts/vectorizer.pickle")
    with open(vectorizer_path, 'rb') as f:
        bow = pickle.load(f)

    bow_df=bow.transform(df_test['preprocess_tweet']).toarray()
    print('feature name==',bow.get_feature_names_out()[:10])
    print('number of uniqe words',bow_df.shape[1])
    print('shape',bow_df.shape)
    bow_train=pd.DataFrame(bow_df)
    bow_train['length_tweet']=df_test['preprocess_tweet'].str.len()
    return bow_train


@app.post("/predict")
def predict(tweet):
    predict = [tweet]
    test = pd.DataFrame(data=[predict],columns=['tweet'])
    to_predict = preprocessed_tweet(test)

    print(to_predict.shape)
    prediction = model.predict(to_predict)
    return {'sentiment' : prediction.tolist()[0]}

out = predict("It's a good day")
print(out)