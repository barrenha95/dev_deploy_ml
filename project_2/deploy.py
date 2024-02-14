import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Loading the model and scaler
model = joblib.load('C:/Users/joao.barrenha/Documents/ds_academy/dev_deploy_ml/project_2/models/model_final.pkl')
scaler = joblib.load('C:/Users/joao.barrenha/Documents/ds_academy/dev_deploy_ml/project_2/models/scaler.pkl')



# Pre-processing function
def preprocess_input(
        Age,
        MonthUse,
        Satisfaction,
        MonthPayment,
        Tier_Basic,
        Tier_Premium,
        Tier_Standard,
        Time_Long,
        Time_Medium,
        Time_Short):
    
    data = pd.DataFrame({
        'Age':          [Age],
        'MonthUse':     [MonthUse],
        'Satisfaction': [Satisfaction],
        'MonthPayment': [MonthPayment],
        'Tier_Basic':   [Tier_Basic],
        'Tier_Premium': [Tier_Premium],
        'Tier_Standard':[Tier_Standard],
        'Time_Long':    [Time_Long],
        'Time_Medium':  [Time_Medium],
        'Time_Short':   [Time_Short],        
    })

    numeric_cols = ['Age',
                    'MonthUse',
                    'Satisfaction',
                    'MonthPayment',
                    'Tier_Basic',
                    'Tier_Premium',
                    'Tier_Standard',
                    'Time_Long',
                    'Time_Medium',
                    'Time_Short'
    ]
    
    data[numeric_cols] = scaler.transform(data[numeric_cols])
    return data

def predict(data):
    prediction = model.predict(data)
    return prediction

## Streamlit ##
st.title("Churn prediction with RandomForest")

# Creation of the fields
Age = st.number_input('Age', min_value = 18, max_value = 100, value = 30)
MonthUse = st.number_input('MonthUse', min_value = 0, max_value = 200, value = 50)
Satisfaction = st.number_input('Satisfaction', min_value = 1, max_value = 5, value = 3)
MonthPayment = st.number_input('MonthPayment', min_value = 0.0, max_value = 500.0, value = 100.0)
tier = st.selectbox('Tier', ['Basic', 'Premium', 'Standard'])
time = st.selectbox('Time of contract', ['Short', 'Medium', 'Long'])

# Prediction button
if st.button('Predict Chrun: '):

    Tier_Basic    = 1 if tier == 'Basic'    else 0
    Tier_Premium  = 1 if tier == 'Premium'  else 0
    Tier_Standard = 1 if tier == 'Standard' else 0

    Time_Short  = 1 if time == 'Short'   else 0
    Time_Medium = 1 if time == 'Medium'  else 0
    Time_Long   = 1 if time == 'Long'    else 0

    input_data = preprocess_input(Age,
                                  MonthUse,
                                  Satisfaction,
                                  MonthPayment,
                                  Tier_Basic,
                                  Tier_Premium,
                                  Tier_Standard,
                                  Time_Long,
                                  Time_Medium,
                                  Time_Short)
    
    st.table(input_data)

    prediction = predict(input_data)

    st.write('Churn? ', 'Yes' if prediction[0] == 1 else 'No')
    st.write('Thank you!')
