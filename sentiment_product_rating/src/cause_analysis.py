import joblib
import pandas as pd

def load_cause_model():
    model = joblib.load('../models/cause_model.pkl')
    mlb = joblib.load('../models/cause_mlb.pkl')
    return model, mlb

def predict_causes(text):
    model, mlb = load_cause_model()
    pred = model.predict([text])
    labels = mlb.inverse_transform(pred)
    return list(labels[0]) if labels else []
