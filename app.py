import streamlit as st
import pickle
import string
import re
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import json


with open('chat_words.json') as f:
    chat_words = json.load(f)

stopwords_eng = stopwords.words('english')

punc = string.punctuation

def preprocessing(text):
    text = re.sub(re.compile('<.*?> '),'',text)
    text = re.sub('[0-90-9]','',text)
    for char in punc:
        text = text.replace(char, '')
    text = text.lower()
    text = [ word for word in word_tokenize(text) if word.lower() not in stopwords_eng]
    text = [ chat_words[word.upper()].lower() if word.upper() in chat_words else word for word in text]
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]
    return ' '.join(text)



with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('tfidf_rf.pkl', 'rb') as f:
    tfidf_rf = pickle.load(f)

# Collect user input
    
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap');

        .title {
            font-family: 'Lobster', cursive;
            font-size: 65px;
            font-weight: 10; /* Set font weight to normal */
            color: #FFFFFF; /* Set color to white */
            text-align: center;
            margin-top: -50px;
        }
    </style>
    <h1 class="title">Sentiment Analysis</h1>
    """, unsafe_allow_html=True)

st.header("Enter Text to Analyze")
input_text = st.text_input(' ')
input_text = preprocessing(input_text)

st.header('Processed Text')
st.write(input_text)

# Check if the input is not empty
if input_text:
    # Transform the input text using the TF-IDF vectorizer
    input_vector = tfidf.transform([input_text])

    # Make prediction using the trained Random Forest model
    output = tfidf_rf.predict(input_vector)

    if output == 0:
        output = 'Negative'
        text_color = 'red'
    else:
        output = 'Positive'
        text_color = 'green'
    # Display the output
    st.header('Sentiment')
    # Display the output with increased text size and appropriate color
    st.write(f"<h1 style='color: {text_color};'>{output}</h1>", unsafe_allow_html=True)
