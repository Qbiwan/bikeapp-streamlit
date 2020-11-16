import pickle

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


# Sidebar

st.sidebar.header("Select input values")

def user_input_features():
    temperature = st.sidebar.slider("temperature", -2.2, 35.1, 17.8)
    humidity = st.sidebar.slider("humidity", 21.3, 75.6, 56.5)
    windspeed = st.sidebar.slider("windspeed", 5.3, 35.2, 17.2)

    data = {
            "temperature": temperature,                
            "humidity": humidity,
            "windspeed": windspeed,
    }

    return pd.DataFrame(data, index=[0])    

df = user_input_features()

st.sidebar.markdown(
""" 
****   
[Download training data](https://raw.githubusercontent.com/cambridgecoding/machinelearningregression/master/data/bikes.csv)
"""
)


# Main

st.header("Bike App")
st.subheader("How many bikes will be on the road today?")
st.write("\n")
st.write("\n")
st.write("You have selected the following inputs:")
st.write(df)

with open("./model/xgbmodel.pkl", "rb") as f:
    load_clf = pickle.load(f)
preds = load_clf.predict(df)[0]

st.write("\n")
st.write("\n")
st.subheader("Prediction by XgBoost Model")
st.write(f"The predicted number of bikes today is {int(preds)}")

