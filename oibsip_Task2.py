# DABHI DHRUVI R
# TASK 2
# UNemployment Analysis
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Unemployment Analysis")
data = pd.read_csv('Unemployment_in_ndia_1.csv')
st.subheader('View Csv File data')
st.write(data.head())

# Check If ANy Null VAlues is in?
st.subheader("Check If Any Null value?")
st.write(data.isnull().sum())

# IF yes then drop this

st.subheader(data.dropna(inplace=True))
st.subheader('data shape')
st.write(data.shape)

Region = data['Region']
Date = data['Date']
Frequency = data['Frequency']
EstimatedLabourParticipationRate = data['EstimatedLabourParticipationRate']
# Estimated_Employed  = data['Estimated_Employed ']
EstimatedLabourParticipationRate = data['EstimatedLabourParticipationRate']
Area = data['Area']

# valuess = ["Region","Date","Frequency","EstimatedLabourParticipationRate","Estimated_Employed","Area"]


# Region 
st.subheader('Estimated Labour Participation Rate by Region')
plt.figure(figsize=(10,9))
plt.bar(Region,EstimatedLabourParticipationRate)
plt.title('Estimated Labour Participation Rate by Region')
plt.ylabel('EstimatedLabourParticipationRate')
plt.xlabel('Region')
plt.title('EstimatedLabourParticipationRate')
plt.xticks(rotation=90)
st.pyplot(plt)



# correlation between the features of this dataset
st.write("Correlation Heatmap:")
plt.figure(figsize=(11, 12))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
st.pyplot(plt)
plt.close()


# using the  bar plot comparing Region and Estimated Employed
st.subheader("Comparison of Region and Estimated Employe")
plt.figure(figsize=(14, 8))
sns.barplot(x='Region', y='Estimated_Employed', hue='Area', data=data)
plt.xlabel('Region')
plt.ylabel('Estimated Employed')
plt.xticks(rotation=90)
plt.title('Comparison of Region and Estimated Employed')
plt.legend(title='Area')
st.pyplot(plt)

