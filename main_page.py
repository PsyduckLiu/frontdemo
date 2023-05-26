from scipy.interpolate import CubicSpline
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import altair as alt
import datetime
import os
from wallstreet import Stock, Call, Put
from pandas_datareader import data as wb
from scipy.stats import norm
from MC_new import AsianPriceOption, VanillaOption

os.environ["http_proxy"]="http://127.0.0.1:10809"
os.environ["https_proxy"]="http://127.0.0.1:10809"

tab1, tab2, tab3 = st.tabs(["Parameters", "All Options Data", "Selected Options Data"])

def LoadDates():
    expirations = yf.Ticker(st.session_state['symbol']).options
    st.session_state.expirations = expirations
    
def LoadOptions():
    option_data_all = yf.Ticker(st.session_state['symbol']).option_chain(date=st.session_state.selectedDate)
    calls = option_data_all.calls
    puts = option_data_all.puts
    calls['optionType'] = 'call'
    puts['optionType'] = 'put'
    option_data = pd.concat(objs=[calls, puts], ignore_index=True)
    st.session_state.option_data = option_data

def LoadTrades():
    if (st.session_state.selectedStrike is not None) & (st.session_state.selectedType is not None):
        option_line = st.session_state.option_data.loc[(st.session_state.option_data['strike'] == st.session_state.selectedStrike) & (st.session_state.option_data['optionType'] == st.session_state.selectedType)]
        option_selected = option_line.iloc[0].at['contractSymbol']
        st.session_state.option_price = option_line.iloc[0].at['lastPrice']
        st.session_state.option_selected = option_selected

def Calculate_IV(S0, K, r, T, N, OPtype, price, n_sim):
    # Calculate price when volatility = 0
    sigma_min = 0.0
    # TODO Call MC simulation
    vanilla_option = VanillaOption(S0=S0, K=K, r=r, sigma=sigma_min, T=T, N=N, OPtype=OPtype)
    C_min = vanilla_option.simulate(n_sim)
    # Calculate price when volatility = 3
    # Simulation will not converge with volatility > 3 and steps = 1e6
    sigma_max = 3
    # TODO Call MC simulation
    vanilla_option = VanillaOption(S0=S0, K=K, r=r, sigma=sigma_max, T=T, N=N, OPtype=OPtype)
    C_max = vanilla_option.simulate(n_sim)

    # Inproper price
    if price < C_min:
        return sigma_min
    if price > C_max:
        return sigma_max

    # Binary search
    while abs(sigma_max - sigma_min) > 1e-6:
        sigma = (sigma_min + sigma_max) / 2
        # TODO Call MC simulation
        vanilla_option = VanillaOption(S0=S0, K=K, r=r, sigma=sigma, T=T, N=N, OPtype=OPtype)
        C = vanilla_option.simulate(n_sim)
        if C < price:
            sigma_min = sigma
            C_min = C
        else:
            sigma_max = sigma
            C_max = C
        print("sigma_min:%f, price_min:%f, sigma_max:%f, price_max:%f" % (sigma_min, C_min, sigma_max, C_max))
    return sigma

def Calculation():
    # Get interest rate
    # TODO
    # US Treasury yield 2023/05/25
    # 爬 抄美债利率
    x = [1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 30]
    y = [0.05818, 0.05378, 0.05427, 0.05192, 0.04433, 0.04113, 0.03821, 0.03810, 0.03768, 0.04002]

    # Compute interest rate function
    cs = CubicSpline(x, y) 

    # Get vanilla option data
    st.session_state.S0 = yf.Ticker(st.session_state['symbol']).history(start=datetime.date.today(),end=datetime.date.today()).iloc[0].at['Close'] # present price
    st.session_state.K = st.session_state.selectedStrike  # assume strike prices are the same for vanilla and exotic options
    st.session_state.T = (datetime.datetime.strptime(st.session_state.selectedDate, '%Y-%m-%d').date()-datetime.date.today()).days/365  # assume time to maturity are the same for vanilla and exotic options
    st.session_state.r = cs(st.session_state.T)  # calculate interest rate
    st.session_state.N = 100  # hyperparameter
    st.session_state.OPtype = st.session_state.selectedType  # assume option types are the same for vanilla and exotic options
    st.session_state.vanilla_price = st.session_state.option_price
    st.session_state.n_sim = 1000000 # hyperparameter
    print("data infor:")
    print(st.session_state.S0, st.session_state.K, st.session_state.r, st.session_state.T, st.session_state.N, st.session_state.OPtype, st.session_state.vanilla_price, st.session_state.n_sim)

    # Compute implied volatility
    vol = Calculate_IV(st.session_state.S0, st.session_state.K, st.session_state.r, st.session_state.T, st.session_state.N, st.session_state.OPtype, st.session_state.vanilla_price, st.session_state.n_sim)
    print("implied volatility:",vol)
    st.session_state.vol = vol

def Pricing():
    if st.session_state.selectedOption == "a":
        print(type(st.session_state.N))
        exotic_option = AsianPriceOption(st.session_state.S0, st.session_state.K, st.session_state.r, st.session_state.vol, st.session_state.T, st.session_state.N, st.session_state.OPtype)
        C = exotic_option.simulate(st.session_state.n_sim)
        st.session_state.price = C

tab1.header("Select Parameters")
tab1.text_input(
    "Enter A Underlying Symbol",
    placeholder="QQQ",
    on_change=LoadDates,
    key='symbol'
)

if 'expirations' in st.session_state:
    st.session_state.selectedDate = tab1.selectbox(
        "Select An Expiration Date",
        options=st.session_state.expirations,
        on_change=LoadOptions
    )
    LoadOptions()

if 'option_data' in st.session_state:
    st.session_state.selectedStrike = tab1.selectbox(
        'Select A Strike',
        options=st.session_state.option_data['strike'],
        on_change=LoadTrades
    ) 
    st.session_state.selectedType = tab1.selectbox(
        "Select An Option Type",
        options={"call","put"},
        on_change=LoadTrades
    )
    LoadTrades()
    tab1.button(
        "OK",
        on_click=Calculation
    )

    tab2.header("All Options Data")
    tab2.write(st.session_state.option_data)

if 'option_selected' in st.session_state:
    st.session_state.history_data = yf.download(st.session_state.option_selected,start='2020-1-1')
    tab3.write(st.session_state.history_data)

if 'vol' in st.session_state:
    tab1.write(st.session_state.vol)
    st.session_state.selectedOption = tab1.selectbox(
        "Select An Option Type",
        options={"a","b"},
        on_change=Pricing
    )
    Pricing()

if 'price' in st.session_state:
    tab1.write(st.session_state.price)