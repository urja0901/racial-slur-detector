from fastapi import FastAPI
from pydantic import BaseModel
from sentiment_analysis import infer_model

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to Twitter sentiment analysis inference"}

class SentimentAnalysis(BaseModel):
    sentiment_input: str
 

@app.post("/predict")
async def predict(tf: SentimentAnalysis):
    return infer_model(
       tf.sentiment_input
    )