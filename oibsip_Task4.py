        #DABHI DHRUVI R
        # TASK 4
        # Spam Mail Detection


import streamlit as st
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Download NLTK stopwords
nltk.download('stopwords')
st.title('Email Spam Detection')
# Load the dataset
Data = pd.read_csv('spam.csv')
st.subheader('View Csv Data')
st.write(Data)
# print(Data)

# Data preprocessing
dd = Data.drop(Data.columns[[2, 3, 4]], axis=1, inplace=True)
st.subheader("After Dropping unnecessary Colummns")
st.write(Data)

Data['v2'] = Data['v2'].apply(lambda x: x.replace('\r\n', ' '))


Data.info()
# Initialize the Porter Stemmer
ste = PorterStemmer()
cor = []
stopwords_words = set(stopwords.words('english'))

# Process each and every  email
for i in range(len(Data)):
    word = Data['v2'].iloc[i].lower()
    word = word.translate(str.maketrans('', '', string.punctuation)).split()
    word = [ste.stem(line) for line in word if line not in stopwords_words]
    word = " ".join(word)
    cor.append(word)


vectorizer = CountVectorizer()
x = vectorizer.fit_transform(cor).toarray()
y = Data.v1

# Split into trian and testing part
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
cli = RandomForestClassifier(n_jobs=-1)
cli.fit(x_train, y_train)

# Streamlit uSINg
st.title('Spam Mail Detection')
st.write('Enter an email to check if it is spam or not.')

# Text input for the email
email_user_input = st.text_area('Email Content')

if st.button('Predict'):
    
    email_Check = email_user_input.lower().translate(str.maketrans('', '', string.punctuation)).split()
    email_Check = [ste.stem(line) for line in email_Check if line not in stopwords_words]
    email_Check = " ".join(email_Check)

    # Vectorize the input email
    email_vectorized = vectorizer.transform([email_Check]).toarray()

    # Predict the email
    prediction = cli.predict(email_vectorized)

    # Show the result
    if prediction[0] == 'spam':
        st.write('The email is Spam.')
    else:
        st.write('The email is Not Spam.')

    # Show the model AC
    AC = cli.score(x_test, y_test)
    st.write(f'Model Accuracy: {AC:.2f}')