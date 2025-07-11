import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)
from datetime import datetime

import ta

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser


plt.style.use('classic')

import MetaTrader5 as mt
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta, timezone

from tqdm import tqdm
import contextlib
import os




def prepare_df(df, timeframe, add_indicators):

    assert timeframe in ['1min', '5min', '15min', '4H', '1D']

    if timeframe != '1min':
        df = df.resample(rule = timeframe).agg(
            {'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'CLOSE': 'last',
            'TICK_VOL': 'sum',
            }).dropna()

    df['AVG_PRICE'] = (df['OPEN'] + df['HIGH'] + df['LOW'] + df['CLOSE'])/4

    df['FLAG_INCREASE_CANDLE'] = np.where(df['CLOSE'] >= df['OPEN'], 1, 0)

    df['BODY'] = df.apply(lambda x: max(x['OPEN'], x['CLOSE']) - min(x['OPEN'], x['CLOSE']),
                                    axis = 1)
    df['UPPER_SHADOW'] = df.apply(lambda x: x['HIGH'] - max(x['OPEN'], x['CLOSE']),
                                            axis = 1)
    df['LOWER_SHADOW'] = df.apply(lambda x: min(x['OPEN'], x['CLOSE']) - x['LOW'],
                                            axis = 1)
    df['WHOLE_RANGE'] = df['HIGH'] - df['LOW']

    df['FLAG_LONG_UPPER_SHADOW'] = np.where(df['UPPER_SHADOW'] >= df['BODY'], 1, 0)
    df['FLAG_LONG_LOWER_SHADOW'] = np.where(df['LOWER_SHADOW'] >= df['BODY'], 1, 0)

    df['FLAG_HIGHER_HIGH(20)'] = np.where(df['HIGH'] >= df['HIGH'].shift(20), 1, 0)
    df['FLAG_HIGHER_LOW(20)'] = np.where(df['LOW'] >= df['LOW'].shift(20), 1, 0)


    #Moving average of TICK_VOL
    df['AVG_VOL(50)'] = df['TICK_VOL'].rolling(50).mean()
    df['FLAG_OVER_AVG_VOL(50)'] = np.where(df['TICK_VOL'] >= df['AVG_VOL(50)'], 1, 0)

    df['AVG_VOL(200)'] = df['TICK_VOL'].rolling(200).mean()
    df['FLAG_OVER_AVG_VOL(200)'] = np.where(df['TICK_VOL'] >= df['AVG_VOL(200)'], 1, 0)

    df['FLAG_UPTREND_VOL(20)'] = np.where(df['TICK_VOL'] >= df['TICK_VOL'].shift(20), 1, 0)


    if add_indicators:
        #RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['CLOSE'],
                                                window = 14).rsi()

        df['FLAG_UNDER_30_RSI'] = np.where(df['RSI'] < 30, 1, 0)
        df['FLAG_OVER_70_RSI'] = np.where(df['RSI'] > 70, 1, 0)
        df['FLAG_UPTREND_RSI(20)'] = np.where(df['RSI'] >= df['RSI'].shift(20), 1, 0)
        
        #Bollinger band
        df['BB_UPPER_BAND(50)'] = ta.volatility.BollingerBands(df['CLOSE'], window = 50, window_dev = 2).bollinger_hband()
        df['BB_LOWER_BAND(50)'] = ta.volatility.BollingerBands(df['CLOSE'], window = 50, window_dev = 2).bollinger_lband()
        
        #Exponential moving average
        df['EMA(50)'] = ta.trend.EMAIndicator(df['CLOSE'],
                                                window = 50).ema_indicator()
        df['POSITION_EMA(50)'] = df.apply(lambda x: 1 if x['EMA(50)'] >= x['HIGH']
                                                                    else (2 if x['EMA(50)'] >= max(x['OPEN'], x['CLOSE'])
                                                                    else (3 if x['EMA(50)'] >= min(x['OPEN'], x['CLOSE'])
                                                                    else (4 if x['EMA(50)'] >= x['LOW'] else 5)
                                                                        )),
                                                    axis = 1)


        df['EMA(200)'] = ta.trend.EMAIndicator(df['CLOSE'],
                                                window = 200).ema_indicator()
        df['POSITION_EMA(200)'] = df.apply(lambda x: 1 if x['EMA(200)'] >= x['HIGH']
                                                                    else (2 if x['EMA(200)'] >= max(x['OPEN'], x['CLOSE'])
                                                                    else (3 if x['EMA(200)'] >= min(x['OPEN'], x['CLOSE'])
                                                                    else (4 if x['EMA(200)'] >= x['LOW'] else 5)
                                                                        )),
                                                    axis = 1)

    #returns
    df['Ret(t)'] = 100*(df['CLOSE'] - df['CLOSE'].shift(1))/df['CLOSE'].shift(1)

    return(df)


def plot_df(df, path, open_tab):

    # Assuming df is your DataFrame with columns: 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICK_VOL', 'EMA50', 'EMA200', 'RSI'

    # Define subplot heights and widths
    subplot_heights = [600, 100, 100]  # Adjust these values based on your preferences
    subplot_widths = [1]  # Only one column

    # Create subplot with 3 rows and 1 column
    fig = make_subplots(rows=3,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=('Main Chart', 'TICK_VOL Chart', 'RSI Chart'),
                        row_heights=subplot_heights,
                        column_widths=subplot_widths,
                        vertical_spacing = 0.05,  # Set the spacing between rows
                        )

    # Subplot 1: Candlestick chart with EMA lines
    cd = go.Candlestick(x=df.index,
                    open=df['OPEN'],
                    high=df['HIGH'],
                    low=df['LOW'],
                    close=df['CLOSE'],
                    increasing=dict(line=dict(color='white', width = 2)),  # Adjust the line attributes for increasing candles
                    decreasing=dict(line=dict(color='blue', width = 2)),
                    name='Candlesticks',
                    opacity = 0.5
                    
                    )

    ema50 = go.Scatter(x=df.index,
                             y=df['EMA(50)'],
                             mode='lines',
                             name='EMA50',
                             line=dict(color='red', width = 2))

    ema200 = go.Scatter(x=df.index,
                             y=df['EMA(200)'],
                             mode='lines',
                             name='EMA200',
                             line=dict(color='yellow', width = 2))
    
    bb_upper = go.Scatter(x=df.index,
                             y=df['BB_UPPER_BAND(50)'],
                             mode='lines',
                             name='BB_UPPER_BAND50',
                             line=dict(color='white', width = 1))

    bb_lower = go.Scatter(x=df.index,
                             y=df['BB_LOWER_BAND(50)'],
                             mode='lines',
                             name='BB_LOWER_BAND50',
                             line=dict(color='white', width = 1))

    fig.add_trace(cd, row=1, col=1)
    fig.add_trace(ema50,
                  row=1,
                  col=1)
    fig.add_trace(ema200,
                  row=1,
                  col=1)
    fig.add_trace(bb_upper,
                  row=1,
                  col=1)
    fig.add_trace(bb_lower,
                  row=1,
                  col=1)
    

    # Subplot 2: TICK_VOL bar chart
    vol = go.Bar(x=df.index,
                         y=df['TICK_VOL'],
                         name='TICK_VOL',
                         marker=dict(color='blue'),
                         #width = 0
                         )

    av50 = go.Scatter(x=df.index,
                             y=df['AVG_VOL(50)'],
                             mode='lines',
                             name='AVG_VOL50',
                             line=dict(color='red', width = 2))

    av200 = go.Scatter(x=df.index,
                             y=df['AVG_VOL(200)'],
                             mode='lines',
                             name='AVG_VOL200',
                             line=dict(color='yellow', width = 2))


    fig.add_trace(vol,
                  row=2,
                  col=1)
    fig.add_trace(av50,
                  row=2,
                  col=1)
    fig.add_trace(av200,
                  row=2,
                  col=1)

    # Subplot 3: RSI chart with threshold lines

    rsi = go.Scatter(x=df.index,
                             y=df['RSI'],
                             mode='lines',
                             name='RSI',
                             line=dict(color='mediumpurple', width = 2))

    rsi30 = dict(type='line',
                       x0=df.index.min(),
                       x1=df.index.max(),
                       y0=30,
                       y1=30,
                       line=dict(color='white', width=1, dash='dash'))

    rsi70 = dict(type='line',
                       x0=df.index.min(),
                       x1=df.index.max(),
                       y0=70,
                       y1=70,
                       line=dict(color='white', width=1, dash='dash'))

    fig.add_trace(rsi,
                  row=3,
                  col=1)

    fig.add_shape(rsi30,
                  row=3,
                  col=1)

    fig.add_shape(rsi70, row=3, col=1)


    # Add darker shaded area between| 30 and 70 in the RSI plot
    fig.add_shape(
        type='rect',
        x0=df.index.min(),
        x1=df.index.max(),
        y0=30,
        y1=70,
        fillcolor='rgba(200, 160, 255, 0.2)',  # Light purple color with opacity
        line=dict(color='rgba(255, 255, 255, 0)'),  # Set border color and opacity
        row=3,
        col=1
    )

    # Add slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=False,
                thickness=0.05,  # Adjust the thickness of the slider
                bgcolor='rgba(0,0,0,0.1)',  # Set the background color of the slider
            ),
            type='date',
            ),

        height = 800,
        width = 1300,
        plot_bgcolor='black',  # Transparent background
        paper_bgcolor='black',  # Transparent paper background
        font = dict(color = 'white'),
        legend = dict(x = 1.01, y = 1),

        xaxis3_rangeslider_visible = True,
        xaxis3_rangeslider_thickness = 0.05,
        
    )

    # Fix y-axis range for each subplot
    fig.update_yaxes(autorange = True, 
                     # range=[df['CLOSE'].min(), df['CLOSE'].max()], 
                     row=1, col=1, fixedrange= False)  # Adjust as needed
    fig.update_yaxes(autorange = True, 
                     # range=[0, df['TICK_VOL'].max()], 
                     row=2, col=1, fixedrange= False)  # Adjust as needed
    fig.update_yaxes(autorange = True, 
                     range=[0, 100], 
                     row=3, col=1, 
                     # fixedrange= False
                    )  # Assuming RSI values range from 0 to 100


    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='white',
        gridcolor='grey',
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # Exclude weekends
            # dict(bounds=[15, 9], pattern="hour"),  # hide hours outside of 9:00 - 15:00
            # dict(bounds=[12, 13], pattern="hour"),  # hide hours outside of 12:00 - 13:00
        ]
    )

    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='white',
        gridcolor='grey'
    )

    if path:
        # Write HTML output
        fig.write_html(path)
        url = path
        if open_tab:
            webbrowser.open(url, new=2)  # open in new tab

    return(fig)


# =================================== Trading meta trader 5 support =================================== #


def login_metatrader(acc_config):
    
    # runs MetaTrader5 client
    # MetaTrader5 platform must be installed
    if not mt.initialize(path = acc_config.path,
                login = acc_config.login,
                password = acc_config.password,
                server = acc_config.server
                ):
        return(mt.last_error())
    else:
        # log in to trading account
        mt.login(acc_config.login, acc_config.password, acc_config.server) 
 

def acc_info_rt(df_acc):
    account_info_dict = mt.account_info()._asdict()
    temp = pd.DataFrame(list(account_info_dict.items())).transpose()
    temp.columns = temp.iloc[0, :]
    temp = temp.drop(index = 0)
    temp['updated_at'] = datetime.now()

    df_acc = pd.concat([df_acc, temp], axis = 0)
    return(df_acc.reset_index())


def plot_acc_info_rt(df_acc):
    # Define subplot heights and widths
    subplot_heights = [100, 100, 100]  # Adjust these values based on your preferences
    subplot_widths = [1]  # Only one column

    # Create subplot with 2 rows and 1 column
    fig = make_subplots(rows=3,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=('Balance, Equity', 'Margin, Free Margin', 'Margin level'),
                        row_heights=subplot_heights,
                        column_widths=subplot_widths,
                        vertical_spacing = 0.05,  # Set the spacing between rows,
                        specs=[
                            [{"secondary_y": True}], 
                            [{"secondary_y": True}],
                            [{"secondary_y": True}],
                            ]
                        )

    # Subplot 1: Balance, Profit, Equity
    balance = go.Bar(x=df_acc['updated_at'],
                                y=df_acc['balance'],
                                name='Balance',
                                # width = 200,
                                marker=dict(color='blue') ,
                                text=df_acc['balance'].values,
                                textposition="outside"
                                )

    equity = go.Bar(x=df_acc['updated_at'],
                                y=df_acc['equity'],
                                name='Equity',
                                # width = 200,
                                marker=dict(color='red'),
                                text=df_acc['equity'].values,
                                textposition="outside"
                                )

    profit = go.Scatter(x=df_acc['updated_at'],
                                y=df_acc['profit'],
                                mode='lines+markers+text',
                                name='Profit',
                                line=dict(color='green', width = 5),
                                text = df_acc['profit'].astype(float).round(2).values,
                                textposition = 'top center')
                                

    fig.add_trace(balance, 
                  row=1, 
                  col=1,
                  secondary_y=False
                  )
    
    fig.add_trace(equity,
                  row=1,
                  col=1,
                  secondary_y=False)
    
    fig.add_trace(profit,
                  row=1,
                  col=1,
                  secondary_y = True)
    
    # # Subplot 2: Margin, Free Margin
    margin = go.Scatter(x=df_acc['updated_at'],
                                y=df_acc['margin'],
                                mode='lines+markers+text',
                                name='Margin',
                                line=dict(color='blue', width = 2),
                                text=df_acc['margin'].values,
                                textposition="top center")

    margin_free = go.Scatter(x=df_acc['updated_at'],
                                y=df_acc['margin_free'],
                                mode='lines+markers+text',
                                name='Margin_free',
                                line=dict(color='yellow', width = 2),
                                text=df_acc['margin_free'].values,
                                textposition="top center")


    fig.add_trace(margin, row=2, col=1)
    fig.add_trace(margin_free,
                    row=2,
                    col=1)

    # # Subplot 3: Margin level
    margin_level = go.Bar(x=df_acc['updated_at'],
                                y=df_acc['margin_level'],
                                name='Margin_level',
                                # width = 200,
                                marker=dict(color='red'),
                                text=df_acc['margin_level'].astype(float).round(2).values,
                                textposition="outside"
                                )

    fig.add_trace(margin_level,
                    row=3,
                    col=1)
    
    # Add slider
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=False,
                thickness=0.05,  # Adjust the thickness of the slider
                bgcolor='rgba(0,0,0,0.1)',  # Set the background color of the slider
            ),
            type='date',
            ),

        height = 800,
        width = 1300,
        plot_bgcolor='black',  # Transparent background
        paper_bgcolor='black',  # Transparent paper background
        font = dict(color = 'white'),
        legend = dict(x = 1.01, y = 1),

        barmode='group', bargap=0.30,bargroupgap=0.0
        
    )

    # Fix y-axis range for each subplot
    fig.update_yaxes(
        # autorange = True, 
        range=[df_acc['balance'].values[0]*0.9, df_acc['balance'].values[0]*1.05], 
        row=1, col=1, fixedrange= False, secondary_y = False)  # Adjust as needed
    
    fig.update_yaxes(
        autorange = True, 
        # range=[df_acc['profit'].min() - 5, df_acc['profit'].max() + 5], 
        # row=1, col=1, fixedrange= False, 
        secondary_y = True
        )  # Adjust as needed

    fig.update_yaxes(
        # autorange = True, 
        range=[-10, df_acc[['margin', 'margin_free']].max().max()*1.5], 
        row=2, col=1, fixedrange= False)  # Adjust as needed
    
    fig.update_yaxes(
        range=[0, df_acc['margin_level'].max()*1.5], 
        row=3, col=1, fixedrange= False)  # Adjust as needed


    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='white',
        gridcolor='grey',
    )

    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='white',
        gridcolor='grey'
    )

    return(fig)


def positions_rt(df_positions):
    positions = mt.positions_get()
    temp = pd.DataFrame(positions, columns = positions[0]._asdict().keys())
    temp['time'] = pd.to_datetime(temp['time'], unit = 's')
    temp['time_msc'] = pd.to_datetime(temp['time_msc'], unit = 'ms')
    temp['time_update'] = pd.to_datetime(temp['time_update'], unit = 's')
    temp['time_update_msc'] = pd.to_datetime(temp['time_update_msc'], unit = 'ms')
    temp['VN_time'] = temp['time'] + pd.Timedelta('7 hours')
    temp['VN_time_msc'] = temp['time_msc'] + pd.Timedelta('7 hours')
    temp['VN_time_update'] = temp['time_update'] + pd.Timedelta('7 hours')
    temp['VN_time_update_msc'] = temp['time_update_msc'] + pd.Timedelta('7 hours')
    temp['updated_at'] = datetime.now()
    df_positions = pd.concat([df_positions, temp], axis = 0)
    trading_symbols = list(df_positions['symbol'].unique())
    return(df_positions, trading_symbols)


def plot_positions_rt(trading_symbols, df_positions):
    for symbol in trading_symbols:
        print('='*100)
        print(symbol)
        print('='*100)
        tickets = df_positions[['ticket', 'time', 'price_open', 'sl', 'tp', 'type', 'reason', 'symbol']].drop_duplicates()
        df_price = pd.DataFrame(mt.copy_rates_range(symbol, mt.TIMEFRAME_M1, df_positions['time'].min() - pd.Timedelta('7 hours'), datetime.now()))

        df_price['time'] = pd.to_datetime(df_price['time'], unit = 's')
        df_price.index = df_price['time']
        df_price = df_price[['open', 'high', 'low', 'close', 'tick_volume']]
        df_price.columns = ['open', 'high', 'low', 'close', 'tick_vol']
        df_price.columns = [col.upper() for col in df_price.columns]
        df_price = prepare_df(df_price, timeframe = '5min')

        fig = plot_df(df_price, path = None, open_tab = False)

        buy_tickets = tickets[tickets['type'] == 0]
        sell_tickets = tickets[tickets['type'] == 1]

        for idt in buy_tickets.index:
            fig.add_annotation(
                text=buy_tickets['price_open'][idt],
                x=buy_tickets['time'][idt],  # X-coordinate of the text box
                y=buy_tickets['price_open'][idt],  # Y-coordinate of the text box
                showarrow=True,
                arrowhead = 2,
                arrowwidth = 2,
                arrowcolor = 'green',
                arrowside = 'end',
                opacity = 1,
                ax=0,
                ay=100,
                font=dict(
                family="Arial, sans-serif",
                size=15,
                color="white",
                ),
                bordercolor="green",
                borderwidth=2,
                borderpad=4,
                bgcolor="green",
            )

            sl = dict(type='line',
                        x0=df_price.index.min(),
                        x1=df_price.index.max(),
                        y0=buy_tickets['sl'][idt],
                        y1=buy_tickets['sl'][idt],
                        line=dict(color='red', width=1, dash='dash'))

            tp = dict(type='line',
                        x0=df_price.index.min(),
                        x1=df_price.index.max(),
                        y0=buy_tickets['tp'][idt],
                        y1=buy_tickets['tp'][idt],
                        line=dict(color='green', width=1, dash='dash'))

            fig.add_shape(sl,
                        row=1,
                        col=1)

            fig.add_shape(tp, 
                        row=1, 
                        col=1)

        for idt in sell_tickets.index:
            fig.add_annotation(
                text=sell_tickets['price_open'][idt],
                x=sell_tickets['time'][idt],  # X-coordinate of the text box
                y=sell_tickets['price_open'][idt],  # Y-coordinate of the text box
                showarrow=True,
                arrowhead = 2,
                arrowwidth = 2,
                arrowcolor = 'red',
                arrowside = 'end',
                opacity = 1,
                ax=0,
                ay=-100,
                font=dict(
                family="Arial, sans-serif",
                size=15,
                color="white",
                ),
                bordercolor='red',
                borderwidth=2,
                borderpad=4,
                bgcolor="red",
            )

            sl = dict(type='line',
                        x0=df_price.index.min(),
                        x1=df_price.index.max(),
                        y0=buy_tickets['sl'][idt],
                        y1=buy_tickets['sl'][idt],
                        line=dict(color='red', width=1, dash='dash'))

            tp = dict(type='line',
                        x0=df_price.index.min(),
                        x1=df_price.index.max(),
                        y0=buy_tickets['tp'][idt],
                        y1=buy_tickets['tp'][idt],
                        line=dict(color='green', width=1, dash='dash'))

            fig.add_shape(sl,
                        row=1,
                        col=1)

            fig.add_shape(tp, 
                        row=1, 
                        col=1)
            
        return(fig)
    

# =================================== Web scraping data (news) support =================================== #


def set_up_driver(num_clicks, time_sleep_open, time_sleep_clicks):
    '''
    This function only supports the scraping from this site: "https://www.businesstoday.in/news".
    It may support other sites but hadnot been tested on.
    Includes progress bar.
    '''
    # Setup headless Chrome
    options = Options()
    options.headless = True
    options.add_argument("--log-level=3")  # Only FATAL
    options.add_argument("--disable-logging")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    # Redirect stderr (to hide native logs from Chrome/TensorFlow/C++)
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
        driver = webdriver.Chrome(options=options)


    # Open the news page
    driver.get("https://www.businesstoday.in/news")
    time.sleep(time_sleep_open)  # Allow JS to load

    # Click the "Load More" button multiple times
    for _ in tqdm(range(num_clicks), desc="Loading more articles", unit = 'page'):  # Adjust range for more clicks
        try:
            load_more_button = driver.find_element(By.ID, "load_more")
            driver.execute_script("arguments[0].scrollIntoView();", load_more_button)
            driver.execute_script("arguments[0].click();", load_more_button)
            time.sleep(time_sleep_clicks)  # Wait for new articles to load
        except Exception as e:
            print("No more content or error:", e)
            break
    return(driver)


def extract_article_content(url):
    '''
    This function only supports the scraping from this site: "https://www.businesstoday.in/news".
    It may support other sites but hadnot been tested on.
    '''
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the posted time of the article

    user_section = soup.find('div', class_='userdetail_share_main')
    if not user_section:
        return{"time": None, "content": "❌ Content section not found."}

    li_tag = user_section.find('li')
    if not li_tag:
        return{"time": None, "content": "❌ Content section not found."}
    

    raw_time = li_tag.get_text(strip=True)
    time_str = raw_time.replace("Updated", "").replace("IST", "").replace(",", "").strip()
    
    try:
        dt_naive = datetime.strptime(time_str, "%b %d %Y %I:%M %p")
        IST = timezone(timedelta(hours=5, minutes=30))
        GMT7 = timezone(timedelta(hours=7))
        dt_ist = dt_naive.replace(tzinfo=IST)
        dt_gmt7 = dt_ist.astimezone(GMT7)
    except ValueError:
        dt_naive = None
        dt_gmt7 = None

    
    # Extract the main content of the page
    main_div = soup.find('div', class_='story_witha_main_sec')
    if not main_div:
        return {"time": dt_gmt7, "content": "❌ Content section not found."}

    text_div = main_div.find('div', class_='text-formatted')
    if not text_div:
        return {"time": dt_gmt7, "content": "❌ Text block not found."}
    
    # Get all non-empty <p> tags, skip ones inside ads, embeds
    paragraphs = []
    for p in text_div.find_all('p', recursive=True):
        if p.find_parent(['div', 'iframe'], class_=['ads__container', 'story_ad_container', 'embedcode']):
            continue  # skip ads or embeds
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text)

    
    paragraphs = "\n\n".join(paragraphs)

    return(dt_gmt7, paragraphs)