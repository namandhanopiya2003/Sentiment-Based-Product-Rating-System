import joblib

def predict_sentiment(text, model_path='../models/sentiment_model.pkl'):
    model = joblib.load(model_path)
    cleaned_text = clean_text(text)
    pred = model.predict([cleaned_text])[0]
    prob = max(model.predict_proba([cleaned_text])[0])
    
    rating = map_sentiment_to_rating(pred, prob)
    return pred, prob, rating

def clean_text(text):
    import re
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def map_sentiment_to_rating(sentiment, prob):
    if sentiment == 'negative':
        return 1 if prob > 0.7 else 2
    elif sentiment == 'neutral':
        return 3
    else:
        return 5 if prob > 0.7 else 4

if __name__ == "__main__":
    # sample_text = "This product is really good and works perfectly!"
    # sample_text = "This product is really bad!"
    sample_text = "This product is ok!"
    
    pred, prob, rating = predict_sentiment(sample_text)
    print(f"Sentiment: {pred}, Confidence: {prob:.2f}, Predicted Rating: {rating}")
