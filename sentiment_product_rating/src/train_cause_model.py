# import required libraries
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# loads dataset containing review texts and associated cause labels
df = pd.read_csv('../data/root_cause_suggestions.csv')

# splits cause labels into lists
df['cause_labels'] = df['cause_labels'].apply(lambda x: x.split(';'))

# binarize cause labels for multi-label classification
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['cause_labels'])

# saves the label binarizer for use during inference
joblib.dump(mlb, '../models/cause_mlb.pkl')

# splits data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['review_text'], y, test_size=0.2, random_state=42)

# created a pipeline with tf-idf vectorizer and one-vs-rest logistic regression classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
])

# trained the model on training data
pipeline.fit(X_train, y_train)

# makees predictions on test data
y_pred = pipeline.predict(X_test)
# prints classification report with precision, recall, and f1-score
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# saves the trained model
joblib.dump(pipeline, '../models/cause_model.pkl')
print("Model saved to models/")

