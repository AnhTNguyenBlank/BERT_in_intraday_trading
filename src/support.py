import pandas as pd
import numpy as np

from datetime import datetime

import ta

import matplotlib.pyplot as plt

plt.style.use('classic')

# import MetaTrader5 as mt
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta, timezone

from tqdm import tqdm
import contextlib
import os


def prepare_df(df, timeframe, add_indicators):

    assert timeframe in ['1min', '5min', '15min', '4h', '1D']

    if timeframe != '1min':
        df = df.resample(rule = timeframe).agg(
            {'OPEN': 'first',
            'HIGH': 'max',
            'LOW': 'min',
            'CLOSE': 'last',
            'TICK_VOL': 'sum',
            }).dropna()

    df['AVG_PRICE'] = (df['OPEN'] + df['HIGH'] + df['LOW'] + df['CLOSE'])/4

    df['FLAG_INCREASE_CANDLE'] = np.where(df['CLOSE'] > df['OPEN'], 1, 0)    
    
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
                                                window = 7).rsi()

        df['FLAG_UNDER_30_RSI'] = np.where(df['RSI'] < 30, 1, 0)
        df['FLAG_OVER_70_RSI'] = np.where(df['RSI'] > 70, 1, 0)
        df['FLAG_UPTREND_RSI(20)'] = np.where(df['RSI'] >= df['RSI'].shift(20), 1, 0)
        
        #Bollinger band
        df['BB_UPPER_BAND(50)'] = ta.volatility.BollingerBands(df['CLOSE'], window = 50, window_dev = 2).bollinger_hband()
        df['POSITION_UPPER_BAND(50)'] = df.apply(lambda x: 1 if x['BB_UPPER_BAND(50)'] >= x['HIGH']
                                                                    else (2 if x['BB_UPPER_BAND(50)'] >= max(x['OPEN'], x['CLOSE'])
                                                                    else (3 if x['BB_UPPER_BAND(50)'] >= min(x['OPEN'], x['CLOSE'])
                                                                    else (4 if x['BB_UPPER_BAND(50)'] >= x['LOW'] else 5)
                                                                        )),
                                                    axis = 1)
        
        df['BB_LOWER_BAND(50)'] = ta.volatility.BollingerBands(df['CLOSE'], window = 50, window_dev = 2).bollinger_lband()
        df['POSITION_LOWER_BAND(50)'] = df.apply(lambda x: 1 if x['BB_LOWER_BAND(50)'] >= x['HIGH']
                                                                    else (2 if x['BB_LOWER_BAND(50)'] >= max(x['OPEN'], x['CLOSE'])
                                                                    else (3 if x['BB_LOWER_BAND(50)'] >= min(x['OPEN'], x['CLOSE'])
                                                                    else (4 if x['BB_LOWER_BAND(50)'] >= x['LOW'] else 5)
                                                                        )),
                                                    axis = 1)
        
        
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
    # df['Ret(t)'] = 100*(df['CLOSE'] - df['CLOSE'].shift(1))/df['CLOSE'].shift(1)

    return(df)


def get_session(hour):
    if 0 <= hour < 7:
        return 1
    elif 7 <= hour < 13:
        return 2
    elif 13 <= hour < 21:
        return 3
    else:
        return 4


# =================================== Web scraping data (news) support =================================== #

def set_up_driver(num_clicks, time_sleep_open):
    '''
    This function only supports the scraping from this site: "https://www.businesstoday.in/news".
    It may support other sites but hadnot been tested on.
    Includes progress bar.
    '''
    # Setup headless Chrome
    options = Options()
    options.headless = True
    options.add_argument("--headless=new")
    options.add_argument("--log-level=3")  # Only FATAL
    options.add_argument("--disable-logging")
    options.add_argument("--disable-dev-shm-usage")
    # options.add_argument("--no-sandbox")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    # Redirect stderr (to hide native logs from Chrome/TensorFlow/C++)
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
        driver = webdriver.Chrome(options=options)

    # Open the news page
    driver.get("https://www.businesstoday.in/news")
    time.sleep(time_sleep_open)  # Allow JS to load

    # Click the "Load More" button multiple times
    for _ in tqdm(range(num_clicks), desc="Loading more articles", unit = 'page'):  # Adjust range for more clicks
        load_more_button = driver.find_element(By.ID, "load_more")
        driver.execute_script("arguments[0].scrollIntoView();", load_more_button)
        driver.execute_script("arguments[0].click();", load_more_button)
        
        # Wait until the spinner disappears, no matter how long it takes
        WebDriverWait(driver, timeout=60).until(
            EC.invisibility_of_element_located((By.CLASS_NAME, "circular_loader_container"))
        )

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