# Dabhi dhruvi R
# Task 1
# IRIS data Classification

import pandas as pd
import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the database
# or view file  just for understanding 
data = pd.read_csv('Iris.csv')
print(data)

iris = datasets.load_iris()
print(iris)

# SPlit the dataset
x = iris.data
y = iris.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Initialize the models
linear_regressor = LinearRegression()
Logistic_Regression = LogisticRegression()
svc_model = SVC(probability=True)

# Train the models
linear_regressor.fit(x_train, y_train)
Logistic_Regression.fit(x_train, y_train)
svc_model.fit(x_train, y_train)

# Save this models  lilogsv
pickle.dump(linear_regressor, open('lin_model.pkl', 'wb'))
pickle.dump(Logistic_Regression, open('log_model.pkl', 'wb'))
pickle.dump(svc_model, open('svc_model.pkl', 'wb'))

def classify(num):
    if num < 0.5:
        return 'Setosa'
    elif num < 1.5:
        return 'Versicolor'
    else:
        return 'Virginica'

def main():
    st.title("IRIS data Classification")

    # Side bar user get the option of this 
    st.sidebar.title(" Please select Model ")
    model_choice = st.sidebar.selectbox("Choose the model", ["Linear Regression", "Logistic Regression", "SVC"])

    # Input  from user
    sepal_length = st.slider("Sepal Length", float(x[:, 0].min()), float(x[:, 0].max()), float(x[:, 0].mean()))
    sepal_width = st.slider("Sepal Width", float(x[:, 1].min()), float(x[:, 1].max()), float(x[:, 1].mean()))
    petal_length = st.slider("Petal Length", float(x[:, 2].min()), float(x[:, 2].max()), float(x[:, 2].mean()))
    petal_width = st.slider("Petal Width", float(x[:, 3].min()), float(x[:, 3].max()), float(x[:, 3].mean()))

    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    # As per the user select it show the prediction
    if st.button('Predict'):
        if model_choice == "Linear Regression":
            model = pickle.load(open('lin_model.pkl', 'rb'))
            prediction = model.predict(input_features)
            st.success(f"Predicted class: {classify(prediction[0])}")
        elif model_choice == "Logistic Regression":
            model = pickle.load(open('log_model.pkl', 'rb'))
            prediction = model.predict(input_features)
            st.success(f"Predicted class: {classify(prediction[0])}")
        elif model_choice == "SVC":
            model = pickle.load(open('svc_model.pkl', 'rb'))
            prediction = model.predict(input_features)
            st.success(f"Predicted class: {classify(prediction[0])}")
            


if __name__ == '__main__':
    main()
    
st.title("IRIS data Classification")
# data = pd.read_csv('Unemployment_in_ndia_1.csv')
st.subheader('View Csv File data')
st.write(data.head())
Id = data['Id']
SepalLengthCm = data['SepalLengthCm']
SepalWidthCm = data['SepalWidthCm']
PetalLengthCm = data['PetalLengthCm']
PetalWidthCm = data['PetalWidthCm']
Species = data['Species']
# Correlation HEatmap 
st.write("Correlation Heatmap")

plt.figure(figsize=(11, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
st.pyplot(plt)
plt.close()



plt.plot(Species,SepalLengthCm,marker='.',linestyle='',color='blue',label="SepalLengthCm")
plt.plot(Species,SepalWidthCm,marker='+',linestyle='',color='Green',label="SepalWidthCm")
# plt.plot(Species,PetalLengthCm,marker='+',linestyle='',color='red',label="PetalLengthCm")
plt.plot(Species,PetalWidthCm,marker='^',linestyle='',color='black',label="PetalWidthCm")
plt.title("DATA")
plt.xlabel('Species')
plt.ylabel('Flower')
plt.legend(bbox_to_anchor=(1.0,0.9),labels=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])

st.pyplot(plt)