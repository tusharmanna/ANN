#import sklearn,keras,pandas,nump and treamlit libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import datetime
import streamlit as st

#load the saved model and encoders
model = keras.models.load_model('ann_model.h5')
sc = pickle.load(open('scaler.pkl', 'rb'))
label_encoder_gender = pickle.load(open('label_encoder.pkl', 'rb'))
one_hot_encoder_geo = pickle.load(open('onehot_encoder.pkl', 'rb'))

#build streamlit app
st.title('Customer Churn Prediction')

#user input
st.subheader('Enter Customer Details')
credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=500)
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=3)
balance = st.number_input('Balance', min_value=0, max_value=100000, value=50000)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
EstimatedSalary=st.number_input('Estimated Salary', min_value=0, max_value=100000, value=50000)

#convert input into dataframe
data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [EstimatedSalary]
})

geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()    
geo_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

data = pd.concat([data.reset_index(drop=True), geo_df], axis=1)

#scale the data
X = sc.transform(data)

#make prediction
prediction = model.predict(X)

prediction_prob=prediction[0][0]

#display prediction
st.subheader('Prediction')
if prediction_prob > 0.5:
    st.write('Customer will leave the bank')
else:
    st.write('Customer will not leave the bank')







