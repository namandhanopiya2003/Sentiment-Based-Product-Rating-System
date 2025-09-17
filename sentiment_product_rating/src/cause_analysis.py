# import required libraries
import joblib
import pandas as pd

# loads the trained cause detection model and its label binarizer
def load_cause_model():
    model = joblib.load('../models/cause_model.pkl')
    mlb = joblib.load('../models/cause_mlb.pkl')
    return model, mlb

# predicts dissatisfaction causes from input text
def predict_causes(text):
    model, mlb = load_cause_model()

    # generates prediction for input text
    pred = model.predict([text])
    
    labels = mlb.inverse_transform(pred)
    # returns list of predicted causes
    return list(labels[0]) if labels else []

