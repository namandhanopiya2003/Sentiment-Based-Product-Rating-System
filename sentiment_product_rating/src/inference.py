import joblib

# This function takes a review text and predicts its sentiment and ratings
def predict_sentiment(text, model_path='../models/sentiment_model.pkl'):
    model = joblib.load(model_path)
    cleaned_text = clean_text(text)
    pred = model.predict([cleaned_text])[0]
    prob = max(model.predict_proba([cleaned_text])[0])

    # It converts sentiment and confidence to a rating
    rating = map_sentiment_to_rating(pred, prob)
    return pred, prob, rating

# This function cleans the input text by removing unwanted characters and formatting
def clean_text(text):
    import re
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# This function converts sentiment and confidence into a star rating (1 to 5)
def map_sentiment_to_rating(sentiment, prob):
    if sentiment == 'negative':
        return 1 if prob > 0.7 else 2                  # Strong negative -> 1 star, else 2 stars
    elif sentiment == 'neutral':
        return 3                                       # Neutral always 3 stars
    else:
        return 5 if prob > 0.7 else 4                  # Strong positive -> 5 stars, else 4 stars

if __name__ == "__main__":
    # sample_text = "This product is really good and works perfectly!"
    # sample_text = "This product is really bad!"
    sample_text = "This product is ok!"
    
    pred, prob, rating = predict_sentiment(sample_text)
    print(f"Sentiment: {pred}, Confidence: {prob:.2f}, Predicted Rating: {rating}")

