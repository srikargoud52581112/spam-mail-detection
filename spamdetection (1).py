import streamlit as st
import pickle as pk
import subprocess

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "scikit-learn"])
    from sklearn.feature_extraction.text import TfidfVectorizer


model = pk.load(open('spam_detection_model.pkl', 'rb'))
vectorizer = pk.load(open('vectorizer.pkl', 'rb'))

st.title("spam mail detection")
user_review = st.text_input('Enter your message')

if st.button('Predict'):
    if user_review:
       
        review_tfidf = vectorizer.transform([user_review]).toarray()
        
       
        result = model.predict(review_tfidf)

        
        if result[0] == 1:
            st.write('not spam')
        else:
            st.write('spam')
    else:
        st.warning("Please enter a message.")


        