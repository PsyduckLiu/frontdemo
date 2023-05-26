from scipy.interpolate import CubicSpline
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

tab1, tab2, tab3 = st.tabs(["Parameters", "All Options Data", "Selected Options Data"])

def LoadDates():
    expirations = yf.Ticker(st.session_state['symbol']).options
    st.session_state.expirations = expirations
    print(expirations)
    
def LoadOptions():
    option_data_all = yf.Ticker(st.session_state['symbol']).option_chain(date=st.session_state['expirationDate'])
    calls = option_data_all.calls
    puts = option_data_all.puts
    calls['optionType'] = 'C'
    puts['optionType'] = 'P'
    option_data = pd.concat(objs=[calls, puts], ignore_index=True)
    st.session_state.option_data = option_data

tab1.header("Select Parameters")
tab1.text_input(
    "Enter A Underlying Symbol",
    placeholder="QQQ",
    on_change=LoadDates,
    key='symbol'
)

if 'expirations' in st.session_state:
    tab1.date_input(
        "Enter Expiration Date",
        options=st.session_state.expirations,
        on_change=LoadOptions,
        key='expirationDate'
    )

if 'option_data' in st.session_state:
#     st.session_state.selectedOption = tab1.selectbox(
#         'Which Option Would Select?',
#         options=st.session_state.option_data['contractSymbol'],
#         key='optionSelect',
#     ) 

    tab2.header("All Options Data")
    tab2.write(st.session_state.option_data)