import joblib
import pandas as pd
import numpy as np

def load_suggestion_model():
    model = joblib.load('../models/suggestion_model.pkl')
    mlb = joblib.load('../models/suggestion_mlb.pkl')
    return model, mlb

def predict_suggestions(text):
    model, mlb = load_suggestion_model()
    proba = model.predict_proba([text])[0]
    labels = mlb.classes_
    
    suggestions = []
    for idx, p in enumerate(proba):
        if p > 0.5: 
            suggestions.append((labels[idx], round(p * 100, 1))) 

    return suggestions
