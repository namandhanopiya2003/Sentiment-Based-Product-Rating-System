from flask import Flask, render_template, request
from src.inference import predict_sentiment

# Initialize Flask application
app = Flask(__name__)

# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    # Initialize default variables
    rating = None
    sentiment = None
    confidence = None
    review = ""

    # Check if the form was submitted
    if request.method == 'POST':
        review = request.form['review']
        sentiment, confidence, rating = predict_sentiment(review)
    return render_template('index.html', rating=rating, sentiment=sentiment, confidence=confidence, review=review)

if __name__ == "__main__":
    app.run(debug=True)
