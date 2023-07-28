import streamlit as stre
import nltk


nltk.download('punkt')

from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import pickle as pk

st=PorterStemmer()

stre.title("SMS Spam Prediction")

def text_transform(text):
    text=text.lower()
    text=word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(st.stem(i))
    return " ".join(y)

tfidf=pk.load(open('vectorizer.pkl','rb'))
model=pk.load(open('model.pkl','rb'))

message = stre.text_area("Enter your message:")

if stre.button("Check Message"):
    transformed_msg=text_transform(message)

    vector=tfidf.transform([transformed_msg])
    prediction=model.predict(vector)[0]

    if prediction == 1:
        stre.header("Message is Spam")

    else:
        stre.success("Message is Not spam!")


