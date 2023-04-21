import streamlit as st
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import altair as alt

@st.cache_data
def load_data():
    yf.pdr_override()
    BTC=wb.DataReader('BTC-USD',start='2018-1-1', proxy="http://127.0.0.1:10809")
    BTC.info()
    ETH=wb.DataReader('ETH-USD',start='2018-1-1', proxy="http://127.0.0.1:10809")
    ETH.info()
    price_BTC = BTC['Adj Close']
    price_ETH = ETH['Adj Close']
    return BTC,ETH,price_BTC,price_ETH

data_load_state = st.text('Loading data...')
BTC,ETH,price_BTC,price_ETH = load_data()
data_load_state.text("Done! (using st.cache_data)")

def intro():
    st.write("# Fixed Income Derivatives Risk Calculation System")
    st.sidebar.success("Select a demo above.")

    st.subheader('price_BTC')
    st.write(BTC)
    st.subheader('price_ETH')
    st.write(ETH)

def calculation_demo():
    st.markdown(f"# {list(page_names_to_funcs.keys())[1]}")

    #log return, volatility
    st.subheader('average return and volatility')
    log_ret_BTC = np.log(price_BTC).diff().iloc[1::]
    log_ret_ETH = np.log(price_ETH).diff().iloc[1::]

    string1 = "average return of BTC: " + str(np.mean(log_ret_BTC))
    string2 = "average return of ETH: " + str(np.mean(log_ret_ETH))
    string3 = "volatility of BTC price: " + str(np.std(log_ret_BTC))
    string4 = "volatility of ETH price: " + str(np.std(log_ret_ETH))
    st.markdown(string1)
    st.markdown(string2)
    st.markdown(string3)
    st.markdown(string4)

def plotting_demo():
    st.markdown(f'# {list(page_names_to_funcs.keys())[2]}')

    chart_data = pd.merge(price_BTC,price_ETH,how='inner',on='Date')
    chart_data.rename(columns = {'Date' : 'Date','Adj Close_x' : 'BTC','Adj Close_y' : 'ETH'},inplace=True)
    st.subheader('visualize the price')
    st.line_chart(chart_data)

    st.subheader('visualize the price V2.0')
    fig = plt.figure(figsize = (10,5))
    plt.plot(price_BTC,color = "Red",label = "BTC-USD")
    plt.plot(price_ETH,color = "Blue",label = "ETH-USD")
    plt.legend()
    plt.title("Price of BTC/USD and ETH/USD")
    plt.show()
    st.pyplot(fig)

    #separate 2 y-axis to compare the evolution of BTC and ETH prices
    st.subheader('compare the evolution of BTC and ETH prices')
    fig, ax1 = plt.subplots(figsize = (9,5))
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('BTC', color="Red")
    ax1.plot( price_BTC, color="Red")
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('ETH', color="Blue")  
    ax2.plot(price_ETH, color="Blue")
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  
    plt.show()
    st.pyplot(fig)

    #plot the normalized figures
    st.subheader('Normalized prices of BTC/USD and ETH/USD')
    fig = plt.figure(figsize = (10,5))
    plt.plot((price_BTC-np.mean(BTC['Adj Close']))/np.std(BTC['Adj Close']),color = "Red",label = "Normalized BTC-USD")
    plt.plot((price_ETH-np.mean(ETH['Adj Close']))/np.std(ETH['Adj Close']),color = "Blue",label = "Normalized ETH-USD")
    plt.legend()
    plt.title("Normalized prices of BTC/USD and ETH/USD")
    plt.show()
    st.pyplot(fig)

def BTM_demo():
    st.markdown(f'# {list(page_names_to_funcs.keys())[3]}')
    # Sensitivity to maturity
    st.subheader('Sensitivity to maturity')
    range_T=[1/365,1/12,1/4,1/2,1,2,3]
    option_price = []
    for i in range_T:
        Asian_option = BTM("fixed","C",S0=57830,K=58000,r=0.01,sigma=0.05,T=i,N=10)
        option_price.append(Asian_option)
    fig = plt.figure(figsize=(10,5))
    plt.plot(range_T,option_price)
    plt.xlabel("maturity")
    plt.ylabel("Asian_option_price")
    plt.title("sensitivity to maturity")
    plt.show()
    st.pyplot(fig)

    # Sensitivity to number of steps
    st.subheader('Sensitivity to number of steps')
    range_N=np.array([2,3,4,5,]+[6+j for j in range(15)])
    option_price = []
    for i in range_N:
        # start_time = time.time()
        Asian_option = BTM("fixed","C",S0=57830,K=58000,r=0.01,sigma=0.05,T=1,N=i)
        option_price.append(Asian_option)
        # print("---",i," steps took ","%s seconds ---" % round(time.time() - start_time,2))
    fig = plt.figure(figsize=(10,5))
    plt.plot(range_N,option_price)
    plt.xlabel("number of steps")
    plt.ylabel("Asian_option_price")
    plt.title("sensitivity to number of steps")
    plt.show()
    st.pyplot(fig)

    # sensitivity to spot price
    st.subheader('sensitivity to spot price')
    range_price=np.array([57830*(0.95+0.01*j) for j in range(10)])
    option_price = []
    for i in range_price:
        Asian_option = BTM("fixed","C",S0=i,K=58000,r=0.01,sigma=0.05,T=1,N=10)
        option_price.append(Asian_option)
    fig = plt.figure(figsize=(10,5))
    plt.plot(range_price,option_price)
    plt.xlabel("spot price")
    plt.ylabel("Asian_option_price")
    plt.title("sensitivity to spot price")
    plt.show()
    st.pyplot(fig)

def HW_BTM_demo():
    st.markdown(f'# {list(page_names_to_funcs.keys())[4]}')
    # Sensitivity to maturity
    st.subheader('sensitivity to spot price')
    range_T=[1/365,1/12,1/4,1/2,1,2,3]
    option_price = []
    for i in range_T:
        Asian_option = HW_BTM("fixed","C",S0=57830,K=58000,r=0.01,sigma=0.05,T=i,step=10,M=10)
        option_price.append(Asian_option)
    fig = plt.figure(figsize=(10,5))
    plt.plot(range_T,option_price)
    plt.xlabel("maturity")  
    plt.ylabel("Asian_option_price")
    plt.title("sensitivity to maturity")
    plt.show()
    st.pyplot(fig)

    # Sensitivity to number of steps
    st.subheader('sensitivity to number of steps')
    range_N=np.array([2,3,4,5,]+[6+j for j in range(50)])
    option_price = []
    for i in range_N:
        # start_time = time.time()
        Asian_option = HW_BTM("fixed","P",S0=57830,K=58000,r=0.01,sigma=0.05,T=1,step=i,M=round(i/3.5)+1)
        option_price.append(Asian_option)
        # print("---",i," steps took ","%s seconds ---" % round(time.time() - start_time,2))
    fig = plt.figure(figsize=(10,5))
    plt.plot(range_N,option_price)
    plt.xlabel("number of steps")
    plt.ylabel("Asian_option_price")
    plt.title("sensitivity to number of steps")
    plt.show()
    st.pyplot(fig)

    # Sensitivity to spot and strike prices
    st.subheader('Sensitivity to spot and strike prices')
    range_price=np.array([57830*(0.95+0.01*j)  for j in range(10)])
    option_price = []
    for i in range_price:
        Asian_option = HW_BTM("fixed","P",S0=i,K=58000,r=0.01,sigma=0.05,T=1,step=10,M=4)
        option_price.append(Asian_option)
    fig = plt.figure(figsize=(10,5))
    plt.plot(range_price,option_price)
    plt.xlabel("spot price")
    plt.ylabel("Asian_option_price")
    plt.title("sensitivity to spot price")
    plt.show()
    st.pyplot(fig)

    # sensitivity to strike price
    st.subheader('ensitivity to strike price')
    option_price = []
    for i in range_price:
        Asian_option = HW_BTM("fixed","C",S0=57830,K=i,r=0.01,sigma=0.05,T=1,step=10,M=4)
        option_price.append(Asian_option)
    fig = plt.figure(figsize=(10,5))
    plt.plot(range_price,option_price)
    plt.xlabel("strike price")
    plt.ylabel("Asian_option_price")
    plt.title("sensitivity to strike price")
    plt.show()
    st.pyplot(fig)

def BS_demo():
    st.markdown(f'# {list(page_names_to_funcs.keys())[5]}')

    #   sensitivity to strike price
    st.subheader('ensitivity to strike price')
    range_price=np.array([57830*(0.95+0.01*j)  for j in range(10)])
    option1 = []
    option2 = []
    for i in range_price:
        Asian_option = HW_BTM("fixed","C",S0=57830,K=i,r=0.01,sigma=0.05,T=1,step=10,M=4)
        European_option = BS_option_price(t=0,St=57830,K=i,T=1,r=0.01,sigma=0.05,Type="call")
        option1.append(Asian_option)
        option2.append(European_option)
    fig = plt.figure(figsize=(10,5))
    plt.plot(range_price,option1,color="Red",label="Asian arithmetic call option with fixed strike")
    plt.plot(range_price,option2,color="Blue",label="European call option")
    plt.xlabel("spot price")
    plt.ylabel("option_price")
    plt.legend()
    plt.title("sensitivity to strike price")
    plt.show()
    st.pyplot(fig)

    #   sensitivity to strike price
    st.subheader('ensitivity to strike price')
    option1 = []
    option2 = []
    for i in range_price:
        Asian_option = HW_BTM("fixed","P",S0=57830,K=i,r=0.01,sigma=0.05,T=1,step=10,M=4)
        European_option = BS_option_price(t=0,St=57830,K=i,T=1,r=0.01,sigma=0.05,Type="put")
        option1.append(Asian_option)
        option2.append(European_option)
    fig = plt.figure(figsize=(10,5))
    plt.plot(range_price,option1,color="Red",label="Asian arithmetic put option with fixed strike")
    plt.plot(range_price,option2,color="Blue",label="European put option")
    plt.xlabel("spot price")
    plt.ylabel("option_price")
    plt.legend()
    plt.title("sensitivity to strike price")
    plt.show()
    st.pyplot(fig)

def BTM(strike_type,option_type,S0, K, r, sigma, T, N):
    deltaT = T/ N
    u = np.exp(sigma*np.sqrt(deltaT))
    d = 1 / u
    proba = (np.exp(r*deltaT) - d)/ (u - d)
    St=[S0]
    At=[S0]
    strike=[K]
    # Compute the leaves S_{N, j} and the average price A_{N, j}
    for i in range(N):
        St= [j* u for j in St]+[j * d for j in St]
        At+=At
        strike+=strike
        for x in range(len(At)):
            At[x]= At[x]+St[x]
    At=np.array(At)/(N+1)
    if strike_type == "fixed":
        if option_type == "C":
            payoff = np.maximum(At-np.array(strike), 0)
        else:
            payoff = np.maximum(np.array(strike)-At, 0)
    else:
        if option_type =="C":
            payoff = np.maximum(np.array(St)-At, 0)
        else:
            payoff = np.maximum(At-np.array(St), 0)
    #calculate backward the option prices
    option_price  = payoff
    for i in range(N):
        length = int(len(option_price)/2)
        option_price = proba*option_price[0:length]+(1-proba)*option_price[length::]  
    return option_price[0]

# Modified BTM
def HW_BTM(strike_type,option_type,S0,K,r,sigma,T,step,M):
    
    N=step
    deltaT=T/step
    u = np.exp(sigma*np.sqrt(deltaT))
    d = 1 / u
    proba = (np.exp(r*deltaT) - d)/ (u - d)
    #calculate average price at note (N,J)
    At=[]    
    St=[]
    strike=np.array([K]*M)
    
    for J in range(N+1):
        At_max = np.array([(S0 * u**(j) * d**0) for j in range(N-J)]+[(S0 * u**(N-J) * d**(j)) for j in range(J+1)])
        At_max = np.sum(At_max)/len(At_max)
        At_min = np.array([(S0 * d**(j) * u**0) for j in range(J+1)]+[(S0 * d**(J) * u**(j+1)) for j in range(N-J)])
        At_min = np.sum(At_min)/len(At_min)
        diff=At_max-At_min
        At+=[[At_max-diff/(M-1)*k for k in range(M)]]
        St=np.array([S0* u**(N-J) * d**J]*M)
        if strike_type == "fixed":
            if option_type == "C":
                payoff = np.maximum(At-strike, 0)
            else:
                payoff = np.maximum(strike-At, 0)
        else:
            if option_type =="C":
                payoff = np.maximum(St-At, 0)
            else:
                payoff = np.maximum(At-St, 0)
    At=np.round(At,4)
    payoff=np.round(payoff,4)
               
    for N in range(step-1,-1,-1):
        At_backward=[]
        At_new=[]
        for J in range(N+1):
            At_max = np.array([(S0 * u**(j) * d**0) for j in range(N-J)]+[(S0 * u**(N-J) * d**(j)) for j in range(J+1)])
            At_max = np.sum(At_max)/len(At_max)
            At_min = np.array([(S0 * d**(j) * u**0) for j in range(J+1)]+[(S0 * d**(J) * u**(j+1)) for j in range(N-J)])
            At_min = np.sum(At_min)/len(At_min)
            diff= At_max-At_min
            At_backward+=[[At_max-diff/(M-1)*k for k in range(M)]]
            St_u=np.array([S0* u**(N+1-J) * d**J]*M)
            St_d=np.array([S0* u**(N-J) * d**(J+1)]*M)
            At_new.append(((N+1)*np.array(At_backward[J])+St_u)/(N+2))
            At_new.append(((N+1)*np.array(At_backward[J])+St_d)/(N+2))
            
        At_backward=np.round(At_backward,4)
        At_new=np.round(At_new,4)
        payoff_new=At_new*0
        payoff_backward=At_backward*0
        for i in range(len(At_backward)):
            for j in range(len(At[i])):
                if At[i][j]==At_new[2*i][j]:
                    payoff_new[2*i][j]=payoff[i][j]
                else:
                    x=(At[i][j]-At_new[2*i][j])*payoff[i][j+1]
                    y=(At_new[2*i][j]-At[i][j+1])*payoff[i][j]
                    payoff_new[2*i][j]= (x+y)/(At[i][j]-At[i][j+1])
                
                if At[i+1][j]==At_new[2*i+1][j]:
                    payoff_new[2*i+1][j]=payoff[i+1][j]
                else:
                    x=(At[i+1][j-1]-At_new[2*i+1][j])*payoff[i+1][j]
                    y=(At_new[2*i+1][j]-At[i+1][j])*payoff[i+1][j-1]
                    payoff_new[2*i+1][j]= (x+y)/(At[i+1][j-1]-At[i+1][j])
            payoff_backward[i]=(payoff_new[2*i]*proba+payoff_new[2*i+1]*(1-proba))*np.exp(-r*deltaT)
        At=At_backward
        payoff=np.round(payoff_backward,4)
    option_price = np.average(payoff)
    return option_price

def BS_option_price(t,St,K,T,r,sigma,Type):
    d1=(np.log(St/K)+(r+0.5*sigma**2)*(T-t))/sigma*np.sqrt(T-t)
    d2=d1-sigma*np.sqrt(T-t)
 
    if Type == 'call':
        option_price=St*norm.cdf(d1)-K*np.exp(-r*(T-t))*norm.cdf(d2)

    elif Type == 'put':
        option_price=K*np.exp(-r*(T-t))*norm.cdf(-d2)-St*norm.cdf(-d1)
    return option_price

page_names_to_funcs = {
    "—": intro,
    "Calculation Demo": calculation_demo,
    "Plotting Demo": plotting_demo,
    "BTM Demo": BTM_demo,
    "Modified BTM Demo": HW_BTM_demo,
    "BS Demo": BS_demo
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()

