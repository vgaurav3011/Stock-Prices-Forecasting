import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

def MACD(nifty_50):
    """
        Function to compute the Moving Average Convergence Divergence and plot the same
    """
    # Calculate the exponential moving average over period of 12
    EMA_12 = pd.Series(nifty_50['Close'].ewm(span=12, min_periods=12).mean())
    # Calculate the exponential moving average over period of 26
    EMA_26 = pd.Series(nifty_50['Close'].ewm(span=26, min_periods=26).mean())
    # Compute the MACD which is the difference between the EMA 12 and EMA 26
    MACD = pd.Series(EMA_12 - EMA_26)
    # Compute the signal line which is taken for a 9 day period for the index
    MACD_signal = pd.Series(MACD.ewm(span=9, min_periods=9).mean())

    # Plotting the MACD and the signal line
    fig = make_subplots(rows=2, cols=1)
    # Add the Closing Price ot the Plot
    fig.add_trace(go.Scatter(x=nifty_50.Date, y=nifty_50.Close, name='CLOSE'), row=1, col=1)
    # Add the EMA for period 12 to the plot
    fig.add_trace(go.Scatter(x=nifty_50.Date, y=EMA_12, name='EMA 12'), row=1, col=1)
    # Add the EMA for period 26 to the plot
    fig.add_trace(go.Scatter(x=nifty_50.Date, y=EMA_26, name='EMA 26'), row=1, col=1)
    # Add the MACD and signal lines to the plot
    fig.add_trace(go.Scatter(x=nifty_50.Date, y=MACD, name='MACD'), row=2, col=1)
    fig.add_trace(go.Scatter(x=nifty_50.Date, y=MACD_signal, name='Signal line'), row=2, col=1)
    # Finally display the entire plot constructed!
    # fig.show()
    return fig

def get_RSI(df, column='Close', time_window=14):

    """Function to make the RSI values for a given stock dataframe"""

    # Differential between the Column
    diff = df[column].diff(1)

    # Integrity of the difference values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # We consider the upchange as positive difference, otherwise keep it as zero
    up_chg[diff > 0] = diff[diff > 0]

    down_chg[diff < 0] = diff[diff < 0]

    # We set change of time_window-1 so our decay is alpha=1/time_window.
    up_chg_avg = up_chg.ewm(com=time_window - 1,
                            min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1,
                                min_periods=time_window).mean()

    RS = abs(up_chg_avg / down_chg_avg)
    df['RSI'] = 100 - 100 / (1 + RS)

    return df


def get_MACD(df, column='Close'):

    """Function to get the EMA of 12 and 26"""

    df['EMA-12'] = df[column].ewm(span=12, adjust=False).mean()
    df['EMA-26'] = df[column].ewm(span=26, adjust=False).mean()

    
    df['MACD'] = df['EMA-12'] - df['EMA-26']

    
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    
    df['Histogram'] = df['MACD'] - df['Signal']

    return df


def plot_RSI(df):

    """Return a graph object figure containing the RSI indicator in the specified row."""
    fig = make_subplots(rows=2, cols=1)
    df = get_RSI(df)

    fig.add_trace(go.Scatter(x=df['Date'].iloc[30:],
                             y=df['RSI'].iloc[30:],
                             name='RSI',
                             line=dict(color='silver', width=2)))

    fig.update_yaxes(title_text='RSI')

    # Thresholds = 70% overvalued and 30% undervalued
    for y_pos, color in zip([70, 30], ['Red', 'Green']):
        fig.add_shape(x0=df['Date'].iloc[1],
                      x1=df['Date'].iloc[-1],
                      y0=y_pos,
                      y1=y_pos,
                      type='line',
                      line=dict(color=color, width=2))

    # Add a text box for each line.
    for y_pos, text, color in zip([64, 36], ['Overvalued', 'Undervalued'], ['Red', 'Green']):
        fig.add_annotation(x=df['Date'].iloc[int(df['Date'].shape[0] / 10)],
                           y=y_pos,
                           text=text,
                           font=dict(size=14, color=color),
                           bordercolor=color,
                           borderwidth=1,
                           borderpad=2,
                           bgcolor='lightsteelblue',
                           opacity=0.75,
                           showarrow=False)

    # Update the y-axis limits.
    ymin = 25 if df['RSI'].iloc[30:].min() > 25 else df['RSI'].iloc[30:].min() - 5
    ymax = 75 if df['RSI'].iloc[30:].max() < 75 else df['RSI'].iloc[30:].max() + 5
    fig.update_yaxes(range=[ymin, ymax])

    return fig

def get_trading_strategy(df, column='Close'):

    """Return the Buy/Sell signal on the specified (price) column (Default = 'Adj Close')."""

    buy_list, sell_list = [], []

    flag = False

    for i in range(0, len(df)):

        if df['MACD'].iloc[i] > df['Signal'].iloc[i] and flag == False:

            buy_list.append(df[column].iloc[i])
            sell_list.append(np.nan)
            flag = True

        elif df['MACD'].iloc[i] < df['Signal'].iloc[i] and flag == True:

            buy_list.append(np.nan)
            sell_list.append(df[column].iloc[i])
            flag = False

        else:

            buy_list.append(np.nan)
            sell_list.append(np.nan)

    # Store the buy and sell signals
    df['Buy'] = buy_list
    df['Sell'] = sell_list

    return df

def ohlc_plot(df):
    fig = go.Figure([go.Ohlc(x=df.Date,
                         open=df.Open,
                         high=df.High,
                         low=df.Low,
                         close=df.Close)])
    fig.update(layout_xaxis_rangeslider_visible=False)
    return fig

def candlestick_chart(df, plot_EMAs=True, plot_MACD=True):

    """Function to make the candlestick chart"""
    df = get_MACD(df)
    df = get_trading_strategy(df)
    fig = make_subplots(rows=2, cols=1)
    
    fig.update_layout(
    autosize=False,
    width=1000,
    height=1000)

    fig.add_trace(go.Candlestick(x=df['Date'],
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlestick Chart'))

    # We will create a moving average of 12 days and 26 days and plot the same
    if (plot_EMAs == True):
        fig.add_trace(go.Scatter(x=df['Date'],
                                 y=df['EMA-12'],
                                 name='12-period EMA',
                                 line=dict(color='dodgerblue', width=2)))

        fig.add_trace(go.Scatter(x=df['Date'],
                                 y=df['EMA-26'],
                                 name='26-period EMA',
                                 line=dict(color='whitesmoke', width=2)))

    # We will use the MACD signals to plot buying or selling signals

    if (plot_MACD == True):

        fig.add_trace(go.Scatter(x=df['Date'],
                                 y=df['Buy'],
                                 name='Buy Signal',
                                 mode='markers',
                                 marker_symbol='triangle-up',
                                 marker=dict(size=9),
                                 line=dict(color='Lime')))

        fig.add_trace(go.Scatter(x=df['Date'],
                                 y=df['Sell'],
                                 name='Sell Signal',
                                 mode='markers',
                                 marker_symbol='triangle-down',
                                 marker=dict(size=9, color='Yellow')))

    fig.update_xaxes(rangeslider={'visible': False})
    fig.update_yaxes(title_text='Price ($)')

    return fig

def prepare_test_set(df):
  X = df.drop(columns=["Date", "Close"])
  y = df["Close"]
  date_column = df["Date"]
  X_test, y_test = X, y
  print(X_test)
  return X_test, y_test, date_column

def prepare_test_set_lstm(df):
  X = df.drop(columns=["Date", "Close"])
  y = df["Close"]
  date_column = df["Date"]
  columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'EMA_9',
       'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal'] 
  test_X, test_y = df[columns], df['Close']
    # reshape input to be 3D
  test_X, test_y = np.array(test_X), np.array(test_y)
  test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))

  return test_X, test_y, date_column


def moving_avg_features(df):
  """
      Function to calculate the exponential moving averages and moving averages over different intervals of days
      Input: Dataframe
      Output: Dataframe with new moving average features
  """

  df['EMA_9'] = df['Close'].ewm(9).mean().shift()
  df['EMA_9'] = df['EMA_9'].fillna(0)
  df['SMA_5'] = df['Close'].rolling(5).mean().shift()
  df['SMA_5'] = df['SMA_5'].fillna(0)
  df['SMA_10'] = df['Close'].rolling(10).mean().shift()
  df['SMA_10'] = df['SMA_10'].fillna(0)
  df['SMA_15'] = df['Close'].rolling(15).mean().shift()
  df['SMA_15'] = df['SMA_15'].fillna(0)
  df['SMA_30'] = df['Close'].rolling(30).mean().shift()
  df['SMA_30'] = df['SMA_30'].fillna(0)
  return df

def relative_strength_idx(df, column='Close', time_window=14):

    """Function to make the RSI values for a given stock dataframe"""

    # Differential between the Column
    diff = df[column].diff(1)

    # Integrity of the difference values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # We consider the upchange as positive difference, otherwise keep it as zero
    up_chg[diff > 0] = diff[diff > 0]

    down_chg[diff < 0] = diff[diff < 0]

    # We set change of time_window-1 so our decay is alpha=1/time_window.
    up_chg_avg = up_chg.ewm(com=time_window - 1,
                            min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window - 1,
                                min_periods=time_window).mean()

    RS = abs(up_chg_avg / down_chg_avg)
    df['RSI'] = 100 - 100 / (1 + RS)

    return df

def macd_signal(df):
  """
      Function to compute the MACD signals
        Input: Dataframe
        Output: Dataframe with MACD signal feature
  """
    # MACD signals are calculated over a period of 12 and 26 days respectively
  EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
  EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
    # Trace the momentum of the Moving Averages
  df['MACD'] = pd.Series(EMA_12 - EMA_26)
  df['MACD'] = df['MACD'].fillna(0)
    # Finally generate a signal over a span of 9 days using the differences
  df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())
  df['MACD_signal'] = df['MACD_signal'].fillna(0)
  return df

def add_features(df):
    """
        Function to create new features in the dataframe
    """
    df = moving_avg_features(df)
    df = relative_strength_idx(df).fillna(0)
    df = macd_signal(df)
    # Scale Features
    columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'EMA_9',
       'SMA_5', 'SMA_10', 'SMA_15', 'SMA_30', 'RSI', 'MACD', 'MACD_signal']
    for i in columns:
        df[i] = (df[i] - df[i].mean())/(df[i].std())

    return df

def get_test_data(symbol):
  df = yf.download(symbol, start="2021-11-01", end=None)
  df = df.reset_index()
  return df

def plot_price(date_column, y_pred, y_test):
    plt.plot(date_column, y_pred, color='green', marker='o', linestyle='dashed', label='Predicted Price')
    plt.plot(date_column, y_test, color='red', label='Actual Price')
    plt.title('Prices Prediction')
    plt.xlabel('Dates')
    plt.ylabel('Prices')
    plt.legend()
