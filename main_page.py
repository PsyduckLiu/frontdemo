from scipy.interpolate import CubicSpline
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import altair as alt
import datetime
from wallstreet import Stock, Call, Put
from pandas_datareader import data as wb
from scipy.stats import norm

tab1, tab2, tab3 = st.tabs(["Parameters", "All Options Data", "Selected Options Data"])

def LoadOptions():
    option_data_all = yf.Ticker(st.session_state['symbol']).option_chain(date=st.session_state['expirationDate'])
    calls = option_data_all.calls
    puts = option_data_all.puts
    calls['optionType'] = 'C'
    puts['optionType'] = 'P'
    option_data = pd.concat(objs=[calls, puts], ignore_index=True)
    st.session_state.option_data = option_data

tab1.header("Select Parameters")
tab1.date_input(
    "Enter Expiration Date",
    datetime.date(2022, 5, 26),
    on_change=LoadOptions,
    key='expirationDate'
)
tab1.text_input(
    "Enter A Underlying Symbol",
    placeholder="QQQ",
    on_change=LoadOptions,
    key='symbol'
)

if 'option_data' in st.session_state:
#     st.session_state.selectedOption = tab1.selectbox(
#         'Which Option Would Select?',
#         options=st.session_state.option_data['contractSymbol'],
#         key='optionSelect',
#     ) 

    tab2.header("All Options Data")
    tab2.write(st.session_state.option_data)