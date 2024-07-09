from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = FastAPI(title="Toxic Comments Detector")

# Load the Tfidf and model
tfidf = pickle.load(open("tf_idf.pkt", "rb"))
nb_model = pickle.load(open("toxicity_model.pkt", "rb"))

class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        text = request.text
        # Transform the input to Tfidf vectors
        text_tfidf = tfidf.transform([text]).toarray()
        
        # Predict the class of the input text
        prediction = nb_model.predict(text_tfidf)[0]

        # Map the predicted class to a string
        class_name = "Toxic" if prediction == 1 else "Non-Toxic"

        # Return the prediction in a JSON response
        return {
            "text": text,
            "class": class_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
