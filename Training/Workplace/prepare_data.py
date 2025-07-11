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

    # driver = set_up_driver(num_clicks = 1000, time_sleep_open = 8, time_sleep_clicks = 8)

    # # After loading all articles, parse page source with BeautifulSoup
    # soup = BeautifulSoup(driver.page_source, 'html.parser')
    # driver.quit()

    # # Save to file
    # with open('D:/BERT_in_intraday_trading/Saved_results/page_source.html', 'w', encoding='utf-8') as f:
    #     f.write(str(soup))


    # Reload from file
    with open('D:/BERT_in_intraday_trading/Saved_results/page_source.html', 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')


    stored_data = []

    # Pre-filter relevant links first
    valid_links = [
        link for link in soup.find_all('a')
        if link.string and len(link.string.strip()) > 35 and link.has_attr('href')
    ]

    for link in tqdm(valid_links, desc="Extracting articles' contents", unit = 'article'):
        full_url = link['href']
        time_posted, content = extract_article_content(full_url)
        date_posted = link['href'][-10:]
        title = link.string.strip()

        temp = dict()

        temp['TITLE'] = title
        temp['URL'] = full_url
        temp['DATE_POSTED'] = date_posted
        temp['TIME_POSTED'] = time_posted
        temp['CONTENT'] = content

        stored_data.append(temp)


    with open("D:/BERT_in_intraday_trading/Training/Data/stored_data.pkl", "wb") as f:
        pickle.dump(stored_data, f)

    
    # import pickle

    # # Load the pickle file
    # with open("D:/BERT_in_intraday_trading/Training/Data/stored_data.pkl", "rb") as f:
    #     my_loaded_array = pickle.load(f)

    # print(my_loaded_array)