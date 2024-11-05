import streamlit as st
import joblib
import re
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

download("wordnet")
download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r'[^a-z ]', " ", text)
    text = re.sub(r'\s+[a-zA-Z]\s+', " ", text)
    text = re.sub(r'\^[a-zA-Z]\s+', " ", text)
    text = re.sub(r'\s+', " ", text.lower())
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    return text.split()

def lemmatize_text(words):
    tokens = [lemmatizer.lemmatize(word) for word in words]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 3]
    return tokens

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Sentiment Analysis")

input_text = st.text_input("Enter your text:", placeholder="I hate the school")

if st.button("Classify"):
    if input_text:
        processed_text = preprocess_text(input_text)
        tokens = lemmatize_text(processed_text)
        tokens_joined = " ".join(tokens)

        X = vectorizer.transform([tokens_joined]).toarray()
        prediction = model.predict(X)

        if prediction[0] == 0:
            sentiment = "Negative sentiment"
        else:
            sentiment = "Positive sentiment"

        st.subheader("Classification Result:")
        st.write(f"Text: {input_text}")
        st.write(f"Prediction: {sentiment}")
    else:
        st.warning("Please enter some text for analysis.")
