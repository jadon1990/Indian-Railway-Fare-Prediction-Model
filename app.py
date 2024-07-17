import streamlit as st
import pickle
import pandas as pd
import numpy as np


# To configure the page title and logo
PAGE_CONFIG = {
                "page_title":"Railway Fare Predictor", 
                # "page_icon":image, 
                "layout":"centered", 
                "initial_sidebar_state":"auto"
            }
st.set_page_config(**PAGE_CONFIG)

st.title('Train Ticket Fare Prediction')

# Layout
pipe = pickle.load(open('./pipe.pkl', 'rb'))
data = pd.read_pickle('./rouute.pkl')
stationData = pd.read_pickle('./station.pkl')

stationFrom = stationData['stationName'].sort_values().unique()

col1, col2 =  st.columns(2)
with col1:
    station_from = st.selectbox('Station From', sorted(stationFrom))
    station_from_cd = stationData[stationData['stationName'] == station_from]["stationCode"].values[0]
    st.text(f"Station Code:\t{station_from_cd}")

stationFrom = station_from_cd

allStationTo = data[data['stationCode'] == stationFrom]['stationTo'].unique()
stationTo = stationData.loc[stationData["stationCode"].isin(allStationTo)]["stationName"]


with col2:
    station_To = st.selectbox('Station To', sorted(stationTo))
    station_To_Cd = stationData[stationData['stationName'] == station_To]["stationCode"].values[0]
    st.text(f"Station Code:\t{station_To_Cd}")

if st.button('Predict Fare'):

    input_data = data.loc[(data['stationCode'] == station_from_cd) & (data['stationTo'] == station_To_Cd),['distance', 'journeyTime', 'trainStatus']]
    result = pipe.predict(input_data)
    rupeeSymbol = (u'\u20B9')
    st.subheader(f"Predicted Score is : {rupeeSymbol}{str(round(float(result[0]),2))}")

