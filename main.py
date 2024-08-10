import streamlit as st
import pickle
import string
from nltk.corpus import stopwords

import nltk
#nltk.download('punkt')

nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

st.write('''
SMS SPAM CLASSIFIER 
         
#Enter the sms in the given box and press the predict button for prediction..
''')

#preprocess_fn = pickle.load(open('preprocess_text (1).pkl','rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

vectorizer = pickle.load(open('final_vectorizer_final_3.2.pkl','rb'))
model = pickle.load(open('voting_classifier_final_3.2.pkl','rb'))

input_sms = st.text_area("Enter the sms")



if st.button('predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = vectorizer.transform([transformed_sms])
    vector_input_dense = vector_input.toarray()
    result = model.predict(vector_input_dense)[0]

    if result == 1:
        st.write("This sms is Spam")
    else:
        st.write("This sms is not Spam")

