from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset 
df = pd.read_csv("Dataset/SMSSpamCollection", sep='\t', header=None, names=["label", "message"])

# Encode labels: 'ham' = 0, 'spam' = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'\d+', '', text)                      # remove numbers
    text = re.sub(r'[^\w\s]', '', text)                  # remove punctuation
    text = text.strip()
    return text

df['cleaned'] = df['message'].apply(clean_text)

# 4. Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# 5. Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# 6. Prediction function
def predict_message(msg):
    cleaned_msg = clean_text(msg)
    vectorized_msg = vectorizer.transform([cleaned_msg])
    prediction = model.predict(vectorized_msg)[0]
    return "Spam" if prediction == 1 else "Ham"

# 7. Web Route to render the page
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        message = request.form["message"]
        prediction = predict_message(message)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
