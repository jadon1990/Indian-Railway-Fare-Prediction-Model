import streamlit as st
import sklearn
import pickle
import pandas as pd
import numpy as np

pipe = pickle.load(open('./pipe.pkl', 'rb'))
data = pd.read_pickle('./rouute.pkl')
stationDetails = pd.read_pickle('./station.pkl')

stationFrom = data['stationCode'].unique()
st.title('Train ticket Fare Prediction')

col1, col2 =  st.columns([1,2])
with col1:
    station_from = st.selectbox('Station From', sorted(stationFrom))


stationFrom = station_from

stationTo = data[data['stationCode'] == stationFrom]['stationTo'].unique()

with col2:
    station_To = st.selectbox('Station To', sorted(stationTo))


if st.button('Predict Fare'):

    input_data = data.loc[(data['stationCode'] == station_from) & (data['stationTo'] == station_To),['distance', 'journeyTime', 'trainStatus']]

    result = pipe.predict(input_data)
    st.header("Predicted Score is : " + str(int(result[0])))

