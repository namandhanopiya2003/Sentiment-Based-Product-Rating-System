# import required libraries
import joblib
import pandas as pd
import numpy as np

# loads the trained suggestion model and its label binarizer
def load_suggestion_model():
    model = joblib.load('../models/suggestion_model.pkl')
    mlb = joblib.load('../models/suggestion_mlb.pkl')
    return model, mlb

# predicts suggestions from input text
def predict_suggestions(text):
    model, mlb = load_suggestion_model()

    # gets predicted probabilities for each suggestion label
    proba = model.predict_proba([text])[0]
    labels = mlb.classes_
    
    suggestions = []
    # collects labels with probability above threshold
    for idx, p in enumerate(proba):
        if p > 0.5: 
            suggestions.append((labels[idx], round(p * 100, 1))) 

    # returns list of suggested actions with confidence scores
    return suggestions

