
import sys
sys.path.insert(0, 'D:/BERT_in_intraday_trading')
from src.support import *
import pickle

from requests.exceptions import TooManyRedirects, ReadTimeout, SSLError


if __name__ == '__main__':

    #=========================== Scrape source for news_data ===========================#

    driver = set_up_driver(num_clicks = 10000, time_sleep_open = 5, time_sleep_btw = 1)

    # After loading all articles, parse page source with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    # Save to file
    with open('D:/BERT_in_intraday_trading/Training/Saved_results/page_source_2.html', 'w', encoding='utf-8') as f:
        f.write(str(soup))

    #=========================== Extract news_data from source ===========================#

    # Reload source from file
    with open('D:/BERT_in_intraday_trading/Training/Saved_results/page_source_2.html', 'r', encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'html.parser')

    # Reload stored news data
    with open("D:/BERT_in_intraday_trading/Training/Data/stored_data_2.pkl", "rb") as f:
        stored_data = pickle.load(f)

    # Pre-filter relevant links first
    valid_links = [
        link for link in soup.find_all('a')
        if link.string and len(link.string.strip()) > 35 and link.has_attr('href')
    ]

    for link in tqdm(valid_links[30000:], desc="Extracting articles' contents", unit = 'article'):
        try:
            time.sleep(1)
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

        except (TooManyRedirects, SSLError, ReadTimeout):
            time.sleep(10)
            continue

    with open("D:/BERT_in_intraday_trading/Training/Data/stored_data_2.pkl", "wb") as f:
        pickle.dump(stored_data, f)

    
    # #=========================== Load news_data ===========================#

    # with open("D:/BERT_in_intraday_trading/Training/Data/stored_data_2.pkl", "rb") as f:
    #     news_data = pickle.load(f)
        
    # news_data = [new for new in news_data if new['CONTENT'] != 'content']
    # news_data = pd.DataFrame(news_data)
    # news_data.set_index(keys = 'TIME_POSTED', inplace = True)
    # news_data.index = pd.to_datetime(news_data.index).tz_localize(None)
    # news_data = news_data[~news_data.index.isna()].sort_index()












