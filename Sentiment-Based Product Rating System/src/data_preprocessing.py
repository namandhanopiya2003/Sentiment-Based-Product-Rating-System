import pandas as pd
import re
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    # Keep only necessary columns
    df = df[['Text', 'Score']].dropna()
    
    # Basic text cleaning
    df['cleaned_text'] = df['Text'].apply(clean_text)
    
    # Convert Score into sentiment labels (optional: 1-2 = negative, 3 = neutral, 4-5 = positive)
    df['sentiment'] = df['Score'].apply(lambda x: 'negative' if x <= 2 else ('neutral' if x == 3 else 'positive'))
    
    return df[['cleaned_text', 'sentiment']]

def clean_text(text):
    # Remove html tags, non-alphabetic chars and lowercase
    text = re.sub(r'<.*?>', ' ', text)  # remove html tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove non-alpha
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_train_test_split(df, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['sentiment'], 
                                                        test_size=test_size, random_state=random_state,
                                                        stratify=df['sentiment'])
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_and_clean_data('../data/amazon_reviews.csv')
    print(df['sentiment'].value_counts())
