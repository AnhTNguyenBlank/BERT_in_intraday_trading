import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 500)

import matplotlib.pyplot as plt

plt.style.use('classic')
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_rows', 300)

# %config inlinebackend.figure_format = 'svg'

import sys

sys.path.insert(0, 'D:/BERT_in_intraday_trading')

from src.support import *
from src.backtest import *
from src.models import *

import pickle


if __name__ == '__main__':

    #=========================== Scrape source for news_data ===========================#

    # driver = set_up_driver(num_clicks = 1000, time_sleep_open = 5)

    # # After loading all articles, parse page source with BeautifulSoup
    # soup = BeautifulSoup(driver.page_source, 'html.parser')
    # driver.quit()

    # # Save to file
    # with open('D:/BERT_in_intraday_trading/Training/Saved_results/page_source.html', 'w', encoding='utf-8') as f:
    #     f.write(str(soup))

    #=========================== Extract news_data from source ===========================#

    # Reload from file
    # with open('D:/BERT_in_intraday_trading/Training/Saved_results/page_source.html', 'r', encoding='utf-8') as f:
    #     html = f.read()
    # soup = BeautifulSoup(html, 'html.parser')


    # stored_data = []

    # # Pre-filter relevant links first
    # valid_links = [
    #     link for link in soup.find_all('a')
    #     if link.string and len(link.string.strip()) > 35 and link.has_attr('href')
    # ]

    # for link in tqdm(valid_links[10000:], desc="Extracting articles' contents", unit = 'article'):
    #     full_url = link['href']
    #     time_posted, content = extract_article_content(full_url)
    #     date_posted = link['href'][-10:]
    #     title = link.string.strip()

    #     temp = dict()

    #     temp['TITLE'] = title
    #     temp['URL'] = full_url
    #     temp['DATE_POSTED'] = date_posted
    #     temp['TIME_POSTED'] = time_posted
    #     temp['CONTENT'] = content

    #     stored_data.append(temp)


    # with open("D:/BERT_in_intraday_trading/Training/Data/stored_data.pkl", "wb") as f:
    #     pickle.dump(stored_data, f)

    
    #=========================== Load news_data ===========================#

    with open("D:/BERT_in_intraday_trading/Training/Data/stored_data.pkl", "rb") as f:
        news_data = pickle.load(f)
        
    news_data = [new for new in news_data if new['CONTENT'] != 'content']
    news_data = pd.DataFrame(news_data)
    news_data.set_index(keys = 'TIME_POSTED', inplace = True)
    news_data.index = pd.to_datetime(news_data.index).tz_localize(None)
    news_data = news_data[~news_data.index.isna()].sort_index()

    #=========================== Load price_data ===========================#

    interval_data = pd.read_pickle('D:/BERT_in_intraday_trading/Training/Data/XAUUSDm_M1.pkl')

    interval_data = interval_data.set_index('DATE_TIME')
    interval_data.index = pd.to_datetime(interval_data.index)

    interval_data['DATE'] = pd.to_datetime(interval_data['DATE'])
    interval_data['OPEN'] = interval_data['OPEN']
    interval_data['HIGH'] = interval_data['HIGH']
    interval_data['LOW'] = interval_data['LOW']
    interval_data['CLOSE'] = interval_data['CLOSE']

    df_1_min = prepare_df(df = interval_data, timeframe = '1min', add_indicators = True)


    #=========================== Labelling ===========================#

    news_data['MEAN_BA'] = news_data.apply(lambda x: df_1_min.loc[(df_1_min.index >= x.name - pd.Timedelta(hours = 4)) & (df_1_min.index <= x.name + pd.Timedelta(hours = 4)), 'Ret(t)'].mean(), axis = 1)
    news_data['VAR_BA'] = news_data.apply(lambda x: df_1_min.loc[(df_1_min.index >= x.name - pd.Timedelta(hours = 4)) & (df_1_min.index <= x.name + pd.Timedelta(hours = 4)), 'Ret(t)'].var(), axis = 1)

    news_data['MEAN_B'] = news_data.apply(lambda x: df_1_min.loc[(df_1_min.index >= x.name - pd.Timedelta(hours = 4)) & (df_1_min.index <= x.name), 'Ret(t)'].mean(), axis = 1)
    news_data['VAR_B'] = news_data.apply(lambda x: df_1_min.loc[(df_1_min.index >= x.name - pd.Timedelta(hours = 4)) & (df_1_min.index <= x.name), 'Ret(t)'].var(), axis = 1)

    news_data['MEAN_A'] = news_data.apply(lambda x: df_1_min.loc[(df_1_min.index >= x.name) & (df_1_min.index <= x.name + pd.Timedelta(hours = 4)), 'Ret(t)'].mean(), axis = 1)
    news_data['VAR_A'] = news_data.apply(lambda x: df_1_min.loc[(df_1_min.index >= x.name) & (df_1_min.index <= x.name + pd.Timedelta(hours = 4)), 'Ret(t)'].var(), axis = 1)

    news_data['RATIO_VAR_A_B'] = news_data['VAR_A']/news_data['VAR_B']
    news_data['RATIO_MEAN_A_B'] = news_data['MEAN_A']/news_data['MEAN_B']
    news_data['FLAG_HIGH_RISK'] = news_data['RATIO_VAR_A_B'].apply(lambda x: 1 if x >= news_data['RATIO_VAR_A_B'].quantile(0.75) else 0)


    news_data.to_pickle('D:/BERT_in_intraday_trading/Training/Data/news_data_w_labels.pkl')