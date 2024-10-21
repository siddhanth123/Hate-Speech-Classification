# Hosts the REST API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from keras.models import load_model
import pickle
from train_model import preprocess_text, tokenize_and_pad
import numpy as np

app = FastAPI()

# Load the model and tokenizer
model = load_model('models/hate_speech_model.h5')
with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: TextInput):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(input.text)
        
        # Tokenize and pad the text
        padded_sequence, _ = tokenize_and_pad([preprocessed_text], tokenizer=tokenizer)
        
        # Make prediction
        prediction = model.predict(padded_sequence)
        
        # Convert prediction to label
        label = "hate" if prediction[0][0] > 0.5 else "noHate"
        
        return {"prediction": label, "probability": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Hello, this is the hate speech detection API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
