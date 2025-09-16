import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv('../data/root_cause_suggestions.csv')

df['suggestions'] = df['suggestions'].apply(lambda x: x.split(';'))

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['suggestions'])

joblib.dump(mlb, '../models/suggestion_mlb.pkl')

X_train, X_test, y_train, y_test = train_test_split(df['review_text'], y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

joblib.dump(pipeline, '../models/suggestion_model.pkl')
print("Suggestion model saved.")
