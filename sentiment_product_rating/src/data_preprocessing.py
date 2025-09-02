import pandas as pd
import re
from sklearn.model_selection import train_test_split

# This function loads the CSV file and cleans the data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)                                  # Reads data from the CSV file

    # Keeps only the 'Text' and 'Score' columns and remove rows with missing values
    df = df[['Text', 'Score']].dropna()

    # Cleans the text column by removing unwanted characters and formatting
    df['cleaned_text'] = df['Text'].apply(clean_text)

    # Creates a new column 'sentiment' based on the 'Score'
    # Score 1 or 2 -> negative, 3 -> neutral, 4 or 5 -> positive
    df['sentiment'] = df['Score'].apply(lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive'))
    
    return df[['cleaned_text', 'sentiment']]

# This function cleans a text string by removing HTML tags and special characters
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# This function splits the data into training and testing sets
def get_train_test_split(df, test_size=0.2, random_state=42):
    # Splits text and sentiment into train and test
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['sentiment'], 
                                                        test_size=test_size, random_state=random_state,
                                                        stratify=df['sentiment'])
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_and_clean_data('../data/amazon_reviews.csv')
    print(df['sentiment'].value_counts())

