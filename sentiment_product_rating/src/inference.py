import joblib
import re
import yaml
from cause_analysis import predict_causes
from suggestion_system import predict_suggestions

def predict_sentiment(text, model_path='../models/sentiment_model.pkl'):
    model = joblib.load(model_path)
    cleaned_text = clean_text(text)
    pred = model.predict([cleaned_text])[0]
    prob = max(model.predict_proba([cleaned_text])[0])
    rating = map_sentiment_to_rating(pred, prob)
    return pred, prob, rating

def clean_text(text):
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
    review = "Very poor quality. Not worth the price."
    # review = "The product is okay, not too bad."
    # review = "Absolutely loved the product! Great quality."

    sentiment, confidence, rating = predict_sentiment(review)

    causes = predict_causes(review)

    suggestions = predict_suggestions(review)
    suggestion_texts = [s[0] for s in suggestions]
    confidence_scores = [f"{s[1]}%" for s in suggestions]

    output = {
        "Review": review,
        "Predicted Sentiment": sentiment,
        "Confidence": f"{confidence:.2f}",
        "Predicted Rating": f"{rating} stars",
        "Inferred Issues": causes,
        "Suggested Improvements": suggestion_texts,
        "Confidence Scores": confidence_scores
    }

    print(yaml.dump(output, sort_keys=False, allow_unicode=True))
