# Main Libraries
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from nsepy import get_history
from datetime import date
import technical_indicators as ti
import xgboost as xgb
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

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
from IPython.display import display, HTML
from tensorflow.keras.models import load_model


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

    st.image(Image.open("assets/images/market.jpg"))
    option = st.selectbox('Select the Stock: ',
    ('Infosys', 'Tech Mahindra', 'HCL', 'Wipro', 'Tata Elxsi', 'NIIT', 'TCS', 'Mindtree', 'Justdial'))
    
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
        




elif choices == 'Forecasting Prices':
    st.subheader('Stock Prices Forecasting')
    st.sidebar.success("Explore the ML forecasts!!")

    st.image(Image.open("assets/images/market.jpg"))
    option = st.selectbox('Select the Stock: ',
    ('Infosys', 'Tech Mahindra', 'HCL', 'Wipro', 'Tata Elxsi', 'NIIT', 'TCS', 'Mindtree', 'Justdial'))
    
    st.write('Your Selection: ', option)
    

    if not option:
            pass
    else:

        if option == "Infosys":
            stock_sticker = "INFY"
            df = ti.get_test_data(stock_sticker)            
            df = ti.add_features(df)
            X_test, y_test, date_column = ti.prepare_test_set(df)
            model_xgb = pickle.load(open("assets/models/infy_xgb.pickle.dat", "rb"))
            
            y_pred = model_xgb.predict(X_test)
            fig, ax = plt.subplots(figsize=(15,5))
            fig = (ti.plot_price(date_column, y_pred, y_test))
            st.pyplot(fig)

            st.write(f'__Root Mean Squared Error__ = {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
            st.write(f'__Mean Absolute Percentage Error__ = {np.sqrt(metrics.mean_absolute_percentage_error(y_test, y_pred))}')
            st.subheader("__RSI__")
            fig = ti.plot_RSI(df)
            st.plotly_chart(fig)
            st.subheader("__OHLC Chart__")
            fig = ti.ohlc_plot(df)
            st.plotly_chart(fig)
            st.subheader("__MACD__")
            fig = ti.MACD(df)
            st.plotly_chart(fig)
            st.subheader("__Trading Strategy__")
            fig = ti.candlestick_chart(df)
            st.plotly_chart(fig)
        
        if option == "Tech Mahindra":
            stock_sticker = "TECHM.NS"
            df = ti.get_test_data(stock_sticker)            
            df = ti.add_features(df)
            X_test, y_test, date_column = ti.prepare_test_set(df)
            model_xgb = pickle.load(open("assets/models/techm_xgb.pickle.dat", "rb"))
            
            y_pred = model_xgb.predict(X_test)
            fig, ax = plt.subplots(figsize=(15,5))
            fig = (ti.plot_price(date_column, y_pred, y_test))
            st.pyplot(fig)

            st.write(f'__Root Mean Squared Error__ = {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
            st.write(f'__Mean Absolute Percentage Error__ = {np.sqrt(metrics.mean_absolute_percentage_error(y_test, y_pred))}')
            st.subheader("__RSI__")
            fig = ti.plot_RSI(df)
            st.plotly_chart(fig)
            st.subheader("__OHLC Chart__")
            fig = ti.ohlc_plot(df)
            st.plotly_chart(fig)
            st.subheader("__MACD__")
            fig = ti.MACD(df)
            st.plotly_chart(fig)
            st.subheader("__Trading Strategy__")
            fig = ti.candlestick_chart(df)
            st.plotly_chart(fig)

        if option == "Justdial":
            stock_sticker = "JUSTDIAL.NS"
            df = ti.get_test_data(stock_sticker)            
            df = ti.add_features(df)
            X_test, y_test, date_column = ti.prepare_test_set(df)
            model_xgb = pickle.load(open("assets/models/jd_xgb.pickle.dat", "rb"))
            
            y_pred = model_xgb.predict(X_test)
            fig, ax = plt.subplots(figsize=(15,5))
            fig = (ti.plot_price(date_column, y_pred, y_test))
            st.pyplot(fig)

            st.write(f'__Root Mean Squared Error__ = {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
            st.write(f'__Mean Absolute Percentage Error__ = {np.sqrt(metrics.mean_absolute_percentage_error(y_test, y_pred))}')
            st.subheader("__RSI__")
            fig = ti.plot_RSI(df)
            st.plotly_chart(fig)
            st.subheader("__OHLC Chart__")
            fig = ti.ohlc_plot(df)
            st.plotly_chart(fig)
            st.subheader("__MACD__")
            fig = ti.MACD(df)
            st.plotly_chart(fig)
            st.subheader("__Trading Strategy__")
            fig = ti.candlestick_chart(df)
            st.plotly_chart(fig)

        if option == "HCL":
            stock_sticker = "HCLTECH.NS"
            df = ti.get_test_data(stock_sticker)            
            df = ti.add_features(df)
            X_test, y_test, date_column = ti.prepare_test_set(df)
            model_xgb = pickle.load(open("assets/models/hcl_xgb.pickle.dat", "rb"))
            
            y_pred = model_xgb.predict(X_test)
            fig, ax = plt.subplots(figsize=(15,5))
            fig = (ti.plot_price(date_column, y_pred, y_test))
            st.pyplot(fig)

            st.write(f'__Root Mean Squared Error__ = {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
            st.write(f'__Mean Absolute Percentage Error__ = {np.sqrt(metrics.mean_absolute_percentage_error(y_test, y_pred))}')
            st.subheader("__RSI__")
            fig = ti.plot_RSI(df)
            st.plotly_chart(fig)
            st.subheader("__OHLC Chart__")
            fig = ti.ohlc_plot(df)
            st.plotly_chart(fig)
            st.subheader("__MACD__")
            fig = ti.MACD(df)
            st.plotly_chart(fig)
            st.subheader("__Trading Strategy__")
            fig = ti.candlestick_chart(df)
            st.plotly_chart(fig)

        if option == "Tata Elxsi":
            stock_sticker = "TATAELXSI.NS"
            df = ti.get_test_data(stock_sticker)            
            df = ti.add_features(df)
            X_test, y_test, date_column = ti.prepare_test_set_lstm(df)
            model = load_model("assets/models/telx_model.h5")
            
            y_pred = model.predict(X_test)
            fig, ax = plt.subplots(figsize=(15,5))
            fig = (ti.plot_price(date_column, y_pred, y_test))
            st.pyplot(fig)

            st.write(f'__Root Mean Squared Error__ = {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
            st.write(f'__Mean Absolute Percentage Error__ = {np.sqrt(metrics.mean_absolute_percentage_error(y_test, y_pred))}')
            st.subheader("__RSI__")
            fig = ti.plot_RSI(df)
            st.plotly_chart(fig)
            st.subheader("__OHLC Chart__")
            fig = ti.ohlc_plot(df)
            st.plotly_chart(fig)
            st.subheader("__MACD__")
            fig = ti.MACD(df)
            st.plotly_chart(fig)
            st.subheader("__Trading Strategy__")
            fig = ti.candlestick_chart(df)
            st.plotly_chart(fig)
        
        if option == "Wipro":
            stock_sticker = "WIPRO.NS"
            df = ti.get_test_data(stock_sticker)            
            df = ti.add_features(df)
            X_test, y_test, date_column = ti.prepare_test_set_lstm(df)
            model = load_model("assets/models/wipro_model.h5")
            
            y_pred = model.predict(X_test)
            fig, ax = plt.subplots(figsize=(15,5))
            fig = (ti.plot_price(date_column, y_pred, y_test))
            st.pyplot(fig)

            st.write(f'__Root Mean Squared Error__ = {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
            st.write(f'__Mean Absolute Percentage Error__ = {np.sqrt(metrics.mean_absolute_percentage_error(y_test, y_pred))}')
            st.subheader("__RSI__")
            fig = ti.plot_RSI(df)
            st.plotly_chart(fig)
            st.subheader("__OHLC Chart__")
            fig = ti.ohlc_plot(df)
            st.plotly_chart(fig)
            st.subheader("__MACD__")
            fig = ti.MACD(df)
            st.plotly_chart(fig)
            st.subheader("__Trading Strategy__")
            fig = ti.candlestick_chart(df)
            st.plotly_chart(fig)
        
        if option == "NIIT":
            stock_sticker = "NIITLTD.NS"
            df = ti.get_test_data(stock_sticker)            
            df = ti.add_features(df)
            X_test, y_test, date_column = ti.prepare_test_set_lstm(df)
            model = load_model("assets/models/niit.h5")
            
            y_pred = model.predict(X_test)
            fig, ax = plt.subplots(figsize=(15,5))
            fig = (ti.plot_price(date_column, y_pred, y_test))
            st.pyplot(fig)

            st.write(f'__Root Mean Squared Error__ = {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
            st.write(f'__Mean Absolute Percentage Error__ = {np.sqrt(metrics.mean_absolute_percentage_error(y_test, y_pred))}')
            st.subheader("__RSI__")
            fig = ti.plot_RSI(df)
            st.plotly_chart(fig)
            st.subheader("__OHLC Chart__")
            fig = ti.ohlc_plot(df)
            st.plotly_chart(fig)
            st.subheader("__MACD__")
            fig = ti.MACD(df)
            st.plotly_chart(fig)
            st.subheader("__Trading Strategy__")
            fig = ti.candlestick_chart(df)
            st.plotly_chart(fig)
        
        if option == "TCS":
            stock_sticker = "TCS.NS"
            df = ti.get_test_data(stock_sticker)            
            df = ti.add_features(df)
            X_test, y_test, date_column = ti.prepare_test_set_lstm(df)
            model = load_model("assets/models/tcs.h5")
            
            y_pred = model.predict(X_test)
            fig, ax = plt.subplots(figsize=(15,5))
            fig = (ti.plot_price(date_column, y_pred, y_test))
            st.pyplot(fig)

            st.write(f'__Root Mean Squared Error__ = {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
            st.write(f'__Mean Absolute Percentage Error__ = {np.sqrt(metrics.mean_absolute_percentage_error(y_test, y_pred))}')
            st.subheader("__RSI__")
            fig = ti.plot_RSI(df)
            st.plotly_chart(fig)
            st.subheader("__OHLC Chart__")
            fig = ti.ohlc_plot(df)
            st.plotly_chart(fig)
            st.subheader("__MACD__")
            fig = ti.MACD(df)
            st.plotly_chart(fig)
            st.subheader("__Trading Strategy__")
            fig = ti.candlestick_chart(df)
            st.plotly_chart(fig)

        if option == "Mindtree":
            stock_sticker = "MINDTREE.NS"
            df = ti.get_test_data(stock_sticker)            
            df = ti.add_features(df)
            X_test, y_test, date_column = ti.prepare_test_set_lstm(df)
            model = load_model("assets/models/mt.h5")
            
            y_pred = model.predict(X_test)
            fig, ax = plt.subplots(figsize=(15,5))
            fig = (ti.plot_price(date_column, y_pred, y_test))
            st.pyplot(fig)

            st.write(f'__Root Mean Squared Error__ = {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
            st.write(f'__Mean Absolute Percentage Error__ = {np.sqrt(metrics.mean_absolute_percentage_error(y_test, y_pred))}')
            st.subheader("__RSI__")
            fig = ti.plot_RSI(df)
            st.plotly_chart(fig)
            st.subheader("__OHLC Chart__")
            fig = ti.ohlc_plot(df)
            st.plotly_chart(fig)
            st.subheader("__MACD__")
            fig = ti.MACD(df)
            st.plotly_chart(fig)
            st.subheader("__Trading Strategy__")
            fig = ti.candlestick_chart(df)
            st.plotly_chart(fig)
