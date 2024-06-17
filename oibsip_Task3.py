                    # dabhi Dhruvi R
                    # Car price Prediction
                    #  Task 3



import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

st.title('Car Price Prediction')

# Read csv file
data = pd.read_csv('car_data.csv')
print("--CSV FILE DATA----")
print(data)

categorical_columns = ['Fuel_Type', 'Selling_type', 'Transmission']

# Converting categorical data into numerical using OneHotEncoder ...
categorical = OneHotEncoder(drop='first', sparse=False)
categorical_num = categorical.fit_transform(data[categorical_columns])

cate_df = pd.DataFrame(categorical_num, columns=categorical.get_feature_names_out(categorical_columns))

num = data[['Year', 'Driven_kms']]
x = np.concatenate([num.reset_index(drop=True), cate_df], axis=1)
y = data['Selling_Price'].values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)

# Save the model and encoder[so that values convert into num]
with open('car_price_model.pkl', 'wb') as file:
    pk.dump(model, file)

with open('encoder.pkl', 'wb') as file:
    pk.dump(categorical, file)

with open('feature_names.pkl', 'wb') as file:
    feature_names = ['Year', 'Driven_kms'] + categorical.get_feature_names_out(categorical_columns).tolist()
    pk.dump(feature_names, file)

st.success("Model Saved Successfully")


# Loading  the model 
with open('car_price_model.pkl', 'rb') as file:
    model = pk.load(file)

with open('encoder.pkl', 'rb') as file:
    encoder = pk.load(file)

with open('feature_names.pkl', 'rb') as file:
    feature_names = pk.load(file)

# Streamlit uses
def get_name(car_name):
    return car_name

data['Car_Name'] = data['Car_Name'].apply(get_name)
# get values from csv in my words 
Select_car = st.selectbox('Select Car brand', data['Car_Name'].unique())
st.subheader(f'Selected Car Brand: {Select_car}')

Select_Year = st.slider('Car Manufactured Year', 2003, 2018)
st.subheader(f'Selected Year: {Select_Year}')

Select_Kms = st.slider('Number of Kms Driven', 11, 200000)
st.subheader(f'Selected Driven Kms: {Select_Kms}')

Select_FUEL = st.selectbox('Select Fuel Type', data['Fuel_Type'].unique())
st.subheader(f'Selected Fuel Type: {Select_FUEL}')

Select_Sellers = st.selectbox('Select Seller Type', data['Selling_type'].unique())
st.subheader(f'Selected Seller Type: {Select_Sellers}')

Select_transmission = st.selectbox('Select Transmission', data['Transmission'].unique())
st.subheader(f'Selected Transmission: {Select_transmission}')

if st.button('Predict'):
    try:

        user_input = pd.DataFrame([[Select_Year, Select_Kms, Select_FUEL, Select_Sellers, Select_transmission]],
                                  columns=['Year', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission'])

       
        input_numbers = encoder.transform(user_input[['Fuel_Type', 'Selling_type', 'Transmission']])
        
        
        input_in_numbers = user_input[['Year', 'Driven_kms']].values
        final_Result = np.concatenate((input_in_numbers, input_numbers), axis=1)

        
        final_Result_df = pd.DataFrame(final_Result, columns=feature_names)

        # finally  Make prediction
        prediction = model.predict(final_Result_df)
        
        # Display the prediction result when user click on btn 
        st.write(f'Predicted Car Price: {prediction[0]}')
    except Exception as e:
        st.error(f"An error occurred: {e}")