from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from input_processing import stemming
from pydantic import BaseModel
import datetime as dt
import pickle

import warnings
warnings.filterwarnings('ignore')

class News_Input(BaseModel):
    title: str
    text: str
    subject: str

app = FastAPI(title='Fake News Detector API',
              description='Accurately detecting fake news')

@app.get("/", response_class=PlainTextResponse)
def home():
    return "Welcome! API is working perfectly well. Use /docs to proceed to check if a news is fake or not."

@app.post("/predict")
def pred(news: News_Input):
    title = news.title
    text = news.text
    subject = news.subject

    data = [[title,  text, subject]]

    data['content'] = data['subject'] +' '+data['title']
    port_stem = PorterStemmer()
    data['content'] = data['content'].apply(stemming)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(data)
    data = vectorizer.transform(data)
    
    loaded_model = pickle.load(open('FakeNewsPrediction.pkl', 'rb'))

    prediction = loaded_model.predict(data)

    if prediction == 0:
        return {"Outcome": "Not a fake news!"}
    else:
        return {"Outcome": "Fake News!"}