## 🧠 ABOUT THIS PROJECT ==>

- This is a Python-based Sentiment Analysis system that uses Natural Language Processing (NLP) techniques to classify Amazon product reviews into positive, neutral, or negative sentiments. It also predicts a product rating based on the detected sentiment.

- The project is designed for academic purposes, prototyping real-world applications, and showcasing NLP capabilities. It supports both CLI and optional web-based interaction using Flask.

---

## ⚙ TECHNOLOGIES USED ==>

- **Python**

- **Pandas** (for data handling)

- **Scikit-learn** (for model training and evaluation)

- **Natural Language Toolkit** (NLTK) (for text preprocessing)

- **Pickle** (for saving trained models)

- **Flask** (optional - for web demo interface)

---

## 📁 PROJECT FOLDER STRUCTURE ==>

sentiment_rating_project/<br>
├── data/<br>
│ └── amazon_reviews.csv                                    # Input dataset<br>
│<br>
├── models/<br>
│ └── sentiment_model.pkl                                   # Trained model file<br>
│<br>
├── src/<br>
│ ├── cause_analysis.py                                     # Loads cause model and predicts inferred issues<br>
│ ├── data_preprocessing.py                                 # Data cleaning & preprocessing logic<br>
│ ├── inference.py                                          # Prediction script<br>
│ ├── suggestion_system.py                                  # Loads suggestion model and predicts improvements<br>
│ ├── train_cause_model.py                                  # Script to train dissatisfaction cause detection model<br>
│ ├── train_model.py                                        # Script to train and evaluate the model<br>
│ ├── train_suggestion_model.py                             # Script to train improvement suggestion model<br>
│ └── utils.py                                              # Helper functions<br>
│<br>
├── app/<br>
│ ├── app.py                                                # Flask web application (optional)<br>
│ └── templates/<br>
│ └── index.html                                            # Simple HTML UI<br>
│<br>
├── venv/<br>
│<br>
├── requirements.txt                                        # Python dependencies<br>
└── README.md                                               # Project description and guide

---

## 📝 WHAT EACH FILE DOES ==>

**data/**:
- Contains the dataset file amazon_reviews.csv with product reviews and ratings.

**src/train_model.py**:
- Reads the dataset.
- Preprocesses the text using NLTK.
- Converts reviews to TF-IDF vectors.
- Trains a Logistic Regression model.
- Saves the trained model as sentiment_model.pkl.

**src/inference.py**:
- Loads the trained model.
- Accepts a review input.
- Outputs the predicted sentiment and corresponding product rating.

**models/**:
- Stores the trained model (sentiment_model.pkl) used for predictions.

**app/app.py**:
- Flask backend that provides API for sentiment and rating prediction.
- Can be connected with any frontend like Android app or web app.

**templates/index.html**:
- Optional basic frontend (can be ignored if using only backend).

**requirements.txt**:
- Lists all Python packages needed to run the project.

---

## 🚀 HOW TO RUN ==>

- Open cmd and run following commands ->

# Step 1: Move to the project directory:
cd "D:\sentiment_product_rating"
D:

# Step 2: Create a virtual environment (this step produces no output if successful, but a folder named venv will appear in your project directory):
python -m venv venv

# Step 3: Activate the virtual environment:
venv\Scripts\activate

# Step 4: Install the required dependencies:
pip install -r requirements.txt

# Step 5: Move to the src folder:
cd src

# Step 6: Train the model:
python train_model.py

# Step 7: Run inference on a sample input:
python inference.py

---

## ✨ SAMPLE OUTPUT ==>

📝 Input Review: "Absolutely loved the product, quality was top-notch!"<br>
😄 Sentiment: Positive<br>
🌟 Predicted Rating: 5<br>
📊 Confidence: 1.00

---

## 📬 CONTACT ==>

For questions or feedback, feel free to reach out!

---

