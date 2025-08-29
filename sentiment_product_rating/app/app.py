from flask import Flask, render_template, request
from src.inference import predict_sentiment

# Creates a Flask web application
app = Flask(__name__)

# This is the main page where users can submit their reviews
@app.route('/', methods=['GET', 'POST'])
def home():
    rating = None
    sentiment = None
    confidence = None
    review = ""
    if request.method == 'POST':
        review = request.form['review']
        sentiment, confidence, rating = predict_sentiment(review)

    # Renders the HTML page, showing the results and the entered review
    return render_template('index.html', rating=rating, sentiment=sentiment, confidence=confidence, review=review)

# Starts the Flask app when this script is run directly
if __name__ == "__main__":
    app.run(debug=True)
