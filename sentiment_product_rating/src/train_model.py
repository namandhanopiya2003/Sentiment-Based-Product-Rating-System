import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from data_preprocessing import load_and_clean_data, get_train_test_split

# This function trains the sentiment analysis model and saves it to a file
def train_and_save_model(data_path, model_save_path):
    # Loads and cleans the dataset from the CSV file
    df = load_and_clean_data(data_path)
    # Splits the data into training and testing sets
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    # Build a machine learning pipeline:
    # Step 1: Convert text into numerical features using TF-IDF
    # Step 2: Train a logistic regression model on those features
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    # Trains the model using the training data
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Makes predictions on the test set
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))                 # Shows performance metrics

    # Saves the trained model pipeline to a file
    joblib.dump(pipeline, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_and_save_model('../data/amazon_reviews.csv', '../models/sentiment_model.pkl')
