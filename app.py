import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('model.h5', custom_objects={'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy})

with open('LabelEncoder.pkl','rb') as file:
    le=pickle.load(file)
with open('StandardScaler.pkl','rb') as file:
    scaler=pickle.load(file)
with open('OneHotEncoder.pkl','rb') as file:
    oe=pickle.load(file)

st.title('Customer churn Prediction')

geography = st.selectbox('Geography', oe.categories_[0])
gender = st.selectbox('Gender', le.classes_)
age=st.slider('Age', 18, 92)
balance=st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1]) 
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[le.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
},index=[0])
encode=oe.transform([[geography]]).toarray()
encode_df=pd.DataFrame(encode,columns=oe.get_feature_names_out())

data=pd.concat([input_data,encode_df],axis=1)

scaled_data=scaler.transform(data)

prediction=model.predict(scaled_data)
prediction_proba=prediction[0][0]

if(prediction_proba>0.5):
    st.write('The customer is likely to churn')
else:
    st.write("the customer is not likely to churn")