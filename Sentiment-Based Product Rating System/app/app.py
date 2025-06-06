from flask import Flask, render_template, request
from src.inference import predict_sentiment

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    rating = None
    sentiment = None
    confidence = None
    review = ""
    if request.method == 'POST':
        review = request.form['review']
        sentiment, confidence, rating = predict_sentiment(review)
    return render_template('index.html', rating=rating, sentiment=sentiment, confidence=confidence, review=review)

if __name__ == "__main__":
    app.run(debug=True)
