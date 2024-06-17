        #  DABHI DHRUVI R
        # SALES PREDICTION USING PYTHON
        # Task 5



import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LinearRegression 

st.title("SALES PREDICTION USING PYTHON")
sales_data = pd.read_csv('Sales.csv')
st.subheader('----View Csv File------')
st.write(sales_data)
print(sales_data)


# view first 5 data
print('----Viewing 5 data of the csv file -----')
st.write('----Viewing 5 data of the csv file -----')
st.write(sales_data.head())
print(sales_data.head())

#check if any null data is?
check = sales_data.isnull().sum()
print(check)
st.subheader("Check If any Null Value ?")
st.write(check)

#data info

st.subheader("DATA Information")
print(sales_data.info())
st.write(sales_data.info())

# Drop unnamed column bec it is unes


sales_data.drop(sales_data.columns[[0]],axis=1,inplace=True)
print('after drop the data\n',sales_data)
st.subheader('after drop the data\n')
st.write(sales_data)

# real csv file data visualzing


# see any null data in file
print(sales_data.isnull().sum())
sales = sales_data['Sales']
plt1 = sales_data['TV']
plt2 = sales_data['Radio']
plt3 = sales_data['Newspaper']

plt.figure(figsize=(12,7))
plt.plot(sales,plt1,marker="*",linestyle="",label='Tv')
plt.plot(sales,plt2,marker="s",linestyle="",label='Radio')
plt.plot(sales,plt3,marker="D",linestyle="",label='NewsPaper')
plt.title('Actual Data')
plt.xlabel('Sales')
plt.legend(bbox_to_anchor=(1.0,0.9),labels=['tv','radio','newspepar'])
plt.ylabel('Advertising')
# plt.plot(sales_data['Newspaper'],label='newspaper sales')
st.pyplot(plt)


# Linear Regression
x = sales_data[['TV','Radio','Newspaper']]
y = sales_data['Sales']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
regressor = LinearRegression()

regressor.fit(x_train,y_train) #actually produces the linear eqn for the data

# predicting the test set
y_pred = regressor.predict(x_test)

# A vs P
st.write('-----Predicted vs Actual Sales------')
views = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
st.write(views)


# R score
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)
st.subheader('R-squared score:')
st.write("R-squared score:", r_squared)


# Vis A vs P
st.subheader('Actual vs Predicted Sales')
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred,color='red',edgecolors='k',alpha=0.8)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='skyblue', linewidth=3)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
st.pyplot(plt)
