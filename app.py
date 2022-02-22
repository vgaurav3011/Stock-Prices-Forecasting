# Main Libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from nsepy import get_history
from datetime import date
import technical_indicators as ti


# Preprocessing Module
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

# System Libraries
import time
from PIL import Image
import base64
import sys
from datetime import datetime
import os 
from functools import reduce
import glob
from IPython.display import display, HTML
from tqdm import tqdm_notebook as tqdm
import json


# Visualization Libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"




def configure_plotly_browser_state():
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
            },
          });
        </script>
        '''))


df = pd.DataFrame()
evaluation_metrics = {}

st.title("Stock Price Forecasting for the Common Man")

st.markdown("""
<style>
body {
    color: #fff;
    background-color: #71706E;
}
</style>
    """, unsafe_allow_html=True)
#0A3648
#13393E

menu=["Exploratory Data Analysis",
      "Forecasting Prices"]

choices = st.sidebar.selectbox("Home", menu)

if choices == 'Exploratory Data Analysis':
    st.subheader('Stock Prices Features and Assessment')
    st.sidebar.success("Explore end to end!")

    st.image(Image.open("assets/images/market.gif"))
    option = st.selectbox('Select the Stock: ',
    ('Infosys', 'Tech Mahindra', 'Hexaware Technologies', 'Wipro', 'Tata Elxsi', 'NIIT', 'TCS', 'Mindtree', 'Justdial'))
    
    st.write('Your Selection: ', option)
    
    start_date = st.date_input("Enter starting date: ", value=None, max_value=None, key=None)
    st.write("Your Start Date: ", start_date)

    end_date = st.date_input("Enter ending date: ", value=None, max_value=None, key=None)
    st.write("Your End Date: ", end_date)

    if not option:
            pass
    else:

        if option == "Infosys":
            stock_sticker = "INFY"
            st.subheader("""Daily **closing price** of Infosys (""" + stock_sticker + ")")
            
            # Fetch Data on the sticker
            stock_data = yf.Ticker(stock_sticker)

            # Fetch Historical Data on sticker

            stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
            stock_df["Date"] = stock_df.index

            # Display Line Chart

            st.line_chart(stock_df.Close)
    
            # Volume Chart
            st.subheader("""Daily **volume** for Infosys (""" + stock_sticker + ")")
            st.line_chart(stock_df.Volume)

            st.sidebar.subheader("""Display Additional Information""")
            actions = st.sidebar.checkbox("Stock Actions")
        
            if actions:
            
                st.subheader("""Stock **Dividends and Splits** for """ + stock_sticker)
            
                display_action = (stock_data.actions)
            
                if display_action.empty == True:
                    st.write("No data available at the moment")
            
                else:
                    st.write(display_action)                
            
            # Technical Indicators
            selected_explore = st.selectbox("", options=['Select your Option', 'Stock MACD Chart', 'RSI Chart', 'OHLC Chart', 'Candlestick Chart'], index=0)
            
            if selected_explore == 'Stock MACD Chart':
                st.write(stock_df)
                st.markdown('')
                st.markdown('**Stock MACD**')
                st.markdown('')
                st.markdown('')
                st.write("Select a Date from minimum of ")
                fig = ti.MACD(stock_df)
                st.plotly_chart(fig)

    
            elif selected_explore == 'RSI Chart':
                st.markdown('**RSI Chart**')
                fig = ti.plot_RSI(stock_df)
                st.plotly_chart(fig)

            elif selected_explore == "OHLC Chart":
                # Plotly has an inbuilt OHLC chart visualization that we will use to create our OHLC
                st.markdown("**OHLC Chart**")
                fig = ti.ohlc_plot(stock_df)
                st.plotly_chart(fig)
            
            elif selected_explore == "Candlestick Chart":
                st.markdown("**Candlestick Chart with Buy and Sell Signals**")
                fig = ti.candlestick_chart(stock_df)
                st.plotly_chart(fig)
        
        elif option == "Tech Mahindra":
            stock_sticker = "TECHM.NS"
            st.subheader("""Daily **closing price** of Tech Mahindra (""" + stock_sticker + ")")
            
            # Fetch Data on the sticker
            stock_data = yf.Ticker(stock_sticker)

            # Fetch Historical Data on sticker

            stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
            stock_df["Date"] = stock_df.index

            # Display Line Chart

            st.line_chart(stock_df.Close)
            
    
            # Volume Chart
            st.subheader("""Daily **volume** for Tech Mahindra (""" + stock_sticker + ")")
            st.line_chart(stock_df.Volume)

            st.sidebar.subheader("""Display Additional Information""")
            actions = st.sidebar.checkbox("Stock Actions")
        
            if actions:
            
                st.subheader("""Stock **Dividends and Splits** for """ + stock_sticker)
            
                display_action = (stock_data.actions)
            
                if display_action.empty == True:
                    st.write("No data available at the moment")
            
                else:
                    st.write(display_action)                
            
            # Technical Indicators
            selected_explore = st.selectbox("", options=['Select your Option', 'Stock MACD Chart', 'RSI Chart', 'OHLC Chart', 'Candlestick Chart'], index=0)
            
            if selected_explore == 'Stock MACD Chart':
                st.write(stock_df)
                st.markdown('')
                st.markdown('**Stock MACD**')
                st.markdown('')
                st.markdown('')
                st.write("Select a Date from minimum of ")
                fig = ti.MACD(stock_df)
                st.plotly_chart(fig)

    
            elif selected_explore == 'RSI Chart':
                st.markdown('**RSI Chart**')
                fig = ti.plot_RSI(stock_df)
                st.plotly_chart(fig)

            elif selected_explore == "OHLC Chart":
                # Plotly has an inbuilt OHLC chart visualization that we will use to create our OHLC
                st.markdown("**OHLC Chart**")
                fig = ti.ohlc_plot(stock_df)
                st.plotly_chart(fig)
            
            elif selected_explore == "Candlestick Chart":
                st.markdown("**Candlestick Chart with Buy and Sell Signals**")
                fig = ti.candlestick_chart(stock_df)
                st.plotly_chart(fig)
        
        elif option == "Wipro":
            stock_sticker = "WIT"
            st.subheader("""Daily **closing price** of Wipro (""" + stock_sticker + ")")
            # Fetch Data on the sticker
            stock_df = yf.Ticker(stock_sticker)

            # Fetch Historical Data on sticker

            stock_df = stock_df.history(period='1d', start=start_date, end=end_date)
            stock_df["Date"] = stock_df.index

            # Display Line Chart

            st.line_chart(stock_df.Close)
            
    
            # Volume Chart
            st.subheader("""Daily **volume** for Wipro (""" + stock_sticker + ")")
            st.line_chart(stock_df.Volume)

            st.sidebar.subheader("""Display Additional Information""")
            actions = st.sidebar.checkbox("Stock Actions")
        
            if actions:
            
                st.subheader("""Stock **Dividends and Splits** for """ + stock_sticker)
            
                display_action = (stock_data.actions)
            
                if display_action.empty == True:
                    st.write("No data available at the moment")
            
                else:
                    st.write(display_action)                
            
            # Technical Indicators
            selected_explore = st.selectbox("", options=['Select your Option', 'Stock MACD Chart', 'RSI Chart', 'OHLC Chart', 'Candlestick Chart'], index=0)
            
            if selected_explore == 'Stock MACD Chart':
                st.write(stock_df)
                st.markdown('')
                st.markdown('**Stock MACD**')
                st.markdown('')
                st.markdown('')
                st.write("Select a Date from minimum of ")
                fig = ti.MACD(stock_df)
                st.plotly_chart(fig)

    
            elif selected_explore == 'RSI Chart':
                st.markdown('**RSI Chart**')
                fig = ti.plot_RSI(stock_df)
                st.plotly_chart(fig)

            elif selected_explore == "OHLC Chart":
                # Plotly has an inbuilt OHLC chart visualization that we will use to create our OHLC
                st.markdown("**OHLC Chart**")
                fig = ti.ohlc_plot(stock_df)
                st.plotly_chart(fig)
            
            elif selected_explore == "Candlestick Chart":
                st.markdown("**Candlestick Chart with Buy and Sell Signals**")
                fig = ti.candlestick_chart(stock_df)
                st.plotly_chart(fig)
        
        elif option == "Hexaware Technologies":
            stock_sticker = "HEXAWARE.NS"
            st.subheader("""Daily **closing price** of Hexaware (""" + stock_sticker + ")")
            
            # Fetch Data on the sticker
            stock_data = yf.Ticker(stock_sticker)

            # Fetch Historical Data on sticker

            stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
            stock_df["Date"] = stock_df.index

            # Display Line Chart

            st.line_chart(stock_df.Close)
            
            
    
            # Volume Chart
            st.subheader("""Daily **volume** for Hexaware (""" + stock_sticker + ")")
            st.line_chart(stock_df.Volume)

            st.sidebar.subheader("""Display Additional Information""")
            actions = st.sidebar.checkbox("Stock Actions")
        
            if actions:
            
                st.subheader("""Stock **Dividends and Splits** for """ + stock_sticker)
            
                display_action = (stock_data.actions)
            
                if display_action.empty == True:
                    st.write("No data available at the moment")
            
                else:
                    st.write(display_action)                
            
            # Technical Indicators
            selected_explore = st.selectbox("", options=['Select your Option', 'Stock MACD Chart', 'RSI Chart', 'OHLC Chart', 'Candlestick Chart'], index=0)
            
            if selected_explore == 'Stock MACD Chart':
                st.write(stock_df)
                st.markdown('')
                st.markdown('**Stock MACD**')
                st.markdown('')
                st.markdown('')
                st.write("Select a Date from minimum of ")
                fig = ti.MACD(stock_df)
                st.plotly_chart(fig)

    
            elif selected_explore == 'RSI Chart':
                st.markdown('**RSI Chart**')
                fig = ti.plot_RSI(stock_df)
                st.plotly_chart(fig)

            elif selected_explore == "OHLC Chart":
                # Plotly has an inbuilt OHLC chart visualization that we will use to create our OHLC
                st.markdown("**OHLC Chart**")
                fig = ti.ohlc_plot(stock_df)
                st.plotly_chart(fig)
            
            elif selected_explore == "Candlestick Chart":
                st.markdown("**Candlestick Chart with Buy and Sell Signals**")
                fig = ti.candlestick_chart(stock_df)
                st.plotly_chart(fig)
        
        elif option == "Mindtree":
            stock_sticker = "MINDTREE.NS"
            st.subheader("""Daily **closing price** of MindTree (""" + stock_sticker + ")")
            
            # Fetch Data on the sticker
            stock_data = yf.Ticker(stock_sticker)

            # Fetch Historical Data on sticker

            stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
            stock_df["Date"] = stock_df.index

            # Display Line Chart

            st.line_chart(stock_df.Close)
            
    
            # Volume Chart
            st.subheader("""Daily **volume** for MindTree (""" + stock_sticker + ")")
            st.line_chart(stock_df.Volume)

            st.sidebar.subheader("""Display Additional Information""")
            actions = st.sidebar.checkbox("Stock Actions")
        
            if actions:
            
                st.subheader("""Stock **Dividends and Splits** for """ + stock_sticker)
            
                display_action = (stock_data.actions)
            
                if display_action.empty == True:
                    st.write("No data available at the moment")
            
                else:
                    st.write(display_action)                
            
            # Technical Indicators
            selected_explore = st.selectbox("", options=['Select your Option', 'Stock MACD Chart', 'RSI Chart', 'OHLC Chart', 'Candlestick Chart'], index=0)
            
            if selected_explore == 'Stock MACD Chart':
                st.write(stock_df)
                st.markdown('')
                st.markdown('**Stock MACD**')
                st.markdown('')
                st.markdown('')
                st.write("Select a Date from minimum of ")
                fig = ti.MACD(stock_df)
                st.plotly_chart(fig)

    
            elif selected_explore == 'RSI Chart':
                st.markdown('**RSI Chart**')
                fig = ti.plot_RSI(stock_df)
                st.plotly_chart(fig)

            elif selected_explore == "OHLC Chart":
                # Plotly has an inbuilt OHLC chart visualization that we will use to create our OHLC
                st.markdown("**OHLC Chart**")
                fig = ti.ohlc_plot(stock_df)
                st.plotly_chart(fig)
            
            elif selected_explore == "Candlestick Chart":
                st.markdown("**Candlestick Chart with Buy and Sell Signals**")
                fig = ti.candlestick_chart(stock_df)
                st.plotly_chart(fig)
        
        elif option == "TCS":
            stock_sticker = "TCS.NS"
            st.subheader("""Daily **closing price** of TCS (""" + stock_sticker + ")")
            
            # Fetch Data on the sticker
            stock_data = yf.Ticker(stock_sticker)

            # Fetch Historical Data on sticker

            stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
            stock_df["Date"] = stock_df.index

            # Display Line Chart

            st.line_chart(stock_df.Close)
            
    
            # Volume Chart
            st.subheader("""Daily **volume** for TCS (""" + stock_sticker + ")")
            st.line_chart(stock_df.Volume)

            st.sidebar.subheader("""Display Additional Information""")
            actions = st.sidebar.checkbox("Stock Actions")
        
            if actions:
            
                st.subheader("""Stock **Dividends and Splits** for """ + stock_sticker)
            
                display_action = (stock_data.actions)
            
                if display_action.empty == True:
                    st.write("No data available at the moment")
            
                else:
                    st.write(display_action)                
            
            # Technical Indicators
            selected_explore = st.selectbox("", options=['Select your Option', 'Stock MACD Chart', 'RSI Chart', 'OHLC Chart', 'Candlestick Chart'], index=0)
            
            if selected_explore == 'Stock MACD Chart':
                st.write(stock_df)
                st.markdown('')
                st.markdown('**Stock MACD**')
                st.markdown('')
                st.markdown('')
                st.write("Select a Date from minimum of ")
                fig = ti.MACD(stock_df)
                st.plotly_chart(fig)

    
            elif selected_explore == 'RSI Chart':
                st.markdown('**RSI Chart**')
                fig = ti.plot_RSI(stock_df)
                st.plotly_chart(fig)

            elif selected_explore == "OHLC Chart":
                # Plotly has an inbuilt OHLC chart visualization that we will use to create our OHLC
                st.markdown("**OHLC Chart**")
                fig = ti.ohlc_plot(stock_df)
                st.plotly_chart(fig)
            
            elif selected_explore == "Candlestick Chart":
                st.markdown("**Candlestick Chart with Buy and Sell Signals**")
                fig = ti.candlestick_chart(stock_df)
                st.plotly_chart(fig)
        
        elif option == "Justdial":
            stock_sticker = "JUSTDIAL.NS"
            st.subheader("""Daily **closing price** of Justdial (""" + stock_sticker + ")")
            
            # Fetch Data on the sticker
            stock_data = yf.Ticker(stock_sticker)

            # Fetch Historical Data on sticker

            stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
            stock_df["Date"] = stock_df.index

            # Display Line Chart

            st.line_chart(stock_df.Close)
            
    
            # Volume Chart
            st.subheader("""Daily **volume** for Justdial (""" + stock_sticker + ")")
            st.line_chart(stock_df.Volume)

            st.sidebar.subheader("""Display Additional Information""")
            actions = st.sidebar.checkbox("Stock Actions")
        
            if actions:
            
                st.subheader("""Stock **Dividends and Splits** for """ + stock_sticker)
            
                display_action = (stock_data.actions)
            
                if display_action.empty == True:
                    st.write("No data available at the moment")
            
                else:
                    st.write(display_action)                
            
            # Technical Indicators
            selected_explore = st.selectbox("", options=['Select your Option', 'Stock MACD Chart', 'RSI Chart', 'OHLC Chart', 'Candlestick Chart'], index=0)
            
            if selected_explore == 'Stock MACD Chart':
                st.write(stock_df)
                st.markdown('')
                st.markdown('**Stock MACD**')
                st.markdown('')
                st.markdown('')
                st.write("Select a Date from minimum of ")
                fig = ti.MACD(stock_df)
                st.plotly_chart(fig)

    
            elif selected_explore == 'RSI Chart':
                st.markdown('**RSI Chart**')
                fig = ti.plot_RSI(stock_df)
                st.plotly_chart(fig)

            elif selected_explore == "OHLC Chart":
                # Plotly has an inbuilt OHLC chart visualization that we will use to create our OHLC
                st.markdown("**OHLC Chart**")
                fig = ti.ohlc_plot(stock_df)
                st.plotly_chart(fig)
            
            elif selected_explore == "Candlestick Chart":
                st.markdown("**Candlestick Chart with Buy and Sell Signals**")
                fig = ti.candlestick_chart(stock_df)
                st.plotly_chart(fig)
        
        elif option == "NIIT":
            stock_sticker = "NIITLTD.NS"
            st.subheader("""Daily **closing price** of NIIT (""" + stock_sticker + ")")
            
            # Fetch Data on the sticker
            stock_data = yf.Ticker(stock_sticker)

            # Fetch Historical Data on sticker

            stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
            stock_df["Date"] = stock_df.index

            # Display Line Chart

            st.line_chart(stock_df.Close)
            
    
            # Volume Chart
            st.subheader("""Daily **volume** for NIIT (""" + stock_sticker + ")")
            st.line_chart(stock_df.Volume)

            st.sidebar.subheader("""Display Additional Information""")
            actions = st.sidebar.checkbox("Stock Actions")
        
            if actions:
            
                st.subheader("""Stock **Dividends and Splits** for """ + stock_sticker)
            
                display_action = (stock_data.actions)
            
                if display_action.empty == True:
                    st.write("No data available at the moment")
            
                else:
                    st.write(display_action)                
            
            # Technical Indicators
            selected_explore = st.selectbox("", options=['Select your Option', 'Stock MACD Chart', 'RSI Chart', 'OHLC Chart', 'Candlestick Chart'], index=0)
            
            if selected_explore == 'Stock MACD Chart':
                st.write(stock_df)
                st.markdown('')
                st.markdown('**Stock MACD**')
                st.markdown('')
                st.markdown('')
                st.write("Select a Date from minimum of ")
                fig = ti.MACD(stock_df)
                st.plotly_chart(fig)

    
            elif selected_explore == 'RSI Chart':
                st.markdown('**RSI Chart**')
                fig = ti.plot_RSI(stock_df)
                st.plotly_chart(fig)

            elif selected_explore == "OHLC Chart":
                # Plotly has an inbuilt OHLC chart visualization that we will use to create our OHLC
                st.markdown("**OHLC Chart**")
                fig = ti.ohlc_plot(stock_df)
                st.plotly_chart(fig)
            
            elif selected_explore == "Candlestick Chart":
                st.markdown("**Candlestick Chart with Buy and Sell Signals**")
                fig = ti.candlestick_chart(stock_df)
                st.plotly_chart(fig)
        
        elif option == "":
            stock_sticker = "JUSTDIAL.NS"
            st.subheader("""Daily **closing price** of Justdial (""" + stock_sticker + ")")
            
            # Fetch Data on the sticker
            stock_data = yf.Ticker(stock_sticker)

            # Fetch Historical Data on sticker

            stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
            stock_df["Date"] = stock_df.index

            # Display Line Chart

            st.line_chart(stock_df.Close)
            
    
            # Volume Chart
            st.subheader("""Daily **volume** for Justdial (""" + stock_sticker + ")")
            st.line_chart(stock_df.Volume)

            st.sidebar.subheader("""Display Additional Information""")
            actions = st.sidebar.checkbox("Stock Actions")
        
            if actions:
            
                st.subheader("""Stock **Dividends and Splits** for """ + stock_sticker)
            
                display_action = (stock_data.actions)
            
                if display_action.empty == True:
                    st.write("No data available at the moment")
            
                else:
                    st.write(display_action)                
            
            # Technical Indicators
            selected_explore = st.selectbox("", options=['Select your Option', 'Stock MACD Chart', 'RSI Chart', 'OHLC Chart', 'Candlestick Chart'], index=0)
            
            if selected_explore == 'Stock MACD Chart':
                st.write(stock_df)
                st.markdown('')
                st.markdown('**Stock MACD**')
                st.markdown('')
                st.markdown('')
                st.write("Select a Date from minimum of ")
                fig = ti.MACD(stock_df)
                st.plotly_chart(fig)

    
            elif selected_explore == 'RSI Chart':
                st.markdown('**RSI Chart**')
                fig = ti.plot_RSI(stock_df)
                st.plotly_chart(fig)

            elif selected_explore == "OHLC Chart":
                # Plotly has an inbuilt OHLC chart visualization that we will use to create our OHLC
                st.markdown("**OHLC Chart**")
                fig = ti.ohlc_plot(stock_df)
                st.plotly_chart(fig)
            
            elif selected_explore == "Candlestick Chart":
                st.markdown("**Candlestick Chart with Buy and Sell Signals**")
                fig = ti.candlestick_chart(stock_df)
                st.plotly_chart(fig)
        elif option == "Tata Elxsi":
            stock_sticker = "TATAELXSI.NS"
            st.subheader("""Daily **closing price** of Tata Elxsi (""" + stock_sticker + ")")
            
            # Fetch Data on the sticker
            stock_data = yf.Ticker(stock_sticker)

            # Fetch Historical Data on sticker

            stock_df = stock_data.history(period='1d', start=start_date, end=end_date)
            stock_df["Date"] = stock_df.index

            # Display Line Chart

            st.line_chart(stock_df.Close)
            
    
            # Volume Chart
            st.subheader("""Daily **volume** for Tata Elxsi (""" + stock_sticker + ")")
            st.line_chart(stock_df.Volume)

            st.sidebar.subheader("""Display Additional Information""")
            actions = st.sidebar.checkbox("Stock Actions")
        
            if actions:
            
                st.subheader("""Stock **Dividends and Splits** for """ + stock_sticker)
            
                display_action = (stock_data.actions)
            
                if display_action.empty == True:
                    st.write("No data available at the moment")
            
                else:
                    st.write(display_action)                
            
            # Technical Indicators
            selected_explore = st.selectbox("", options=['Select your Option', 'Stock MACD Chart', 'RSI Chart', 'OHLC Chart', 'Candlestick Chart'], index=0)
            
            if selected_explore == 'Stock MACD Chart':
                st.write(stock_df)
                st.markdown('')
                st.markdown('**Stock MACD**')
                st.markdown('')
                st.markdown('')
                st.write("Select a Date from minimum of ")
                fig = ti.MACD(stock_df)
                st.plotly_chart(fig)

    
            elif selected_explore == 'RSI Chart':
                st.markdown('**RSI Chart**')
                fig = ti.plot_RSI(stock_df)
                st.plotly_chart(fig)

            elif selected_explore == "OHLC Chart":
                # Plotly has an inbuilt OHLC chart visualization that we will use to create our OHLC
                st.markdown("**OHLC Chart**")
                fig = ti.ohlc_plot(stock_df)
                st.plotly_chart(fig)
            
            elif selected_explore == "Candlestick Chart":
                st.markdown("**Candlestick Chart with Buy and Sell Signals**")
                fig = ti.candlestick_chart(stock_df)
                st.plotly_chart(fig)
        




elif choices == 'Train Your Own Drogon (Machine Learning Models)':
    st.subheader('Train Machine Learning Models for Stock Prediction & Generate your own Buy/Sell Signals using the best Model')
    st.sidebar.success("The most valuable commodity I know of is information.")

    
    st.markdown('**_Real_ _Time_ ML Training** for any Stocks')
    st.write('We have created this pipeline for multiple Model training on Multiple stocks at the same time and evaluating them')

    
    st.write('Make sure you have Extracted features for the Stocks you want to train models on using first Tab')
    
    result = glob.glob( '*.csv' )
    #st.write( result )
    stock = []
    for val in result:
        stock.append(val.split('.')[0])
    
    st.markdown('**_Recently_ _Extracted_ Stocks** -')
    st.write(stock[:5])
    cols1 = ['NKE', 'JNJ']
    st.markdown('**_Select_ _Stocks_ _to_ Train**')
    Stocks = st.multiselect("", stock, default=cols1)
    
    options = ['Linear Regression', 'Random Forest', 'XGBoost']
    cols2 = ['Linear Regression', 'Random Forest']
    st.markdown('**_Select_ _Machine_ _Learning_ Algorithms** to Train')
    models = st.multiselect("", options, default=cols2)
    
    
    file = './' + stock[0] + '.csv'
    df_stock = pd.read_csv(file)
    df_stock = df_stock.drop(columns=['Date', 'Date_col'])
    #st.write(df_stock.columns)
    st.markdown('Select from your **_Extracted_ features** or use default')
    st.write('Select all Extracted features')
    all_features = st.checkbox('Select all Extracted features')
    cols = ['Open', 'High', 'Low', 'Close(t)', 'Upper_Band', 'MA200', 'ATR', 'ROC', 'QQQ_Close', 'SnP_Close', 'DJIA_Close', 'DJIA(t-5)']
    if all_features:
        cols = df_stock.columns.tolist()
        cols.pop(len(df_stock.columns)-1)

    features = st.multiselect("", df_stock.columns.tolist(), default=cols)
    
    
    submit = st.button('Train Your DROGON')
    if submit:
        try:
            training = Stock_Prediction_Modeling(Stocks, models, features)
            training.pipeline_sequence()
            with open('./metrics.txt') as f:
                eval_metrics = json.load(f)

            

        except:
            st.markdown('There seems to be a error - **_check_ logs**')
            print("Unexpected error:", sys.exc_info())
            print()

    
        Metrics = pd.DataFrame.from_dict({(i,j): eval_metrics[i][j] 
                               for i in eval_metrics.keys() 
                               for j in eval_metrics[i].keys()},
                           orient='index')

        st.write(Metrics)
    
    