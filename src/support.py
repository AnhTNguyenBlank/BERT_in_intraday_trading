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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from bs4 import BeautifulSoup
import requests
from datetime import datetime, timedelta, timezone

from tqdm import tqdm
import contextlib
import os
from dateutil import parser as dateutil_parser
import pickle
import json
import random


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
## "https://www.businesstoday.in/news"
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

## https://cointelegraph.com/tags/{tag}
def set_up_driver_cointelegraph(tag, num_clicks, time_sleep_open):
    '''
    Loads https://cointelegraph.com/tags/{tag} and clicks Load More.
    '''
    options = Options()
    options.headless = True
    options.add_argument("--headless=new")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-logging")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
        driver = webdriver.Chrome(options=options)

    driver.get(f"https://cointelegraph.com/tags/{tag}")

    time.sleep(time_sleep_open)

    for _ in tqdm(range(num_clicks), desc=f"[{tag}] Loading pages", unit="page"):
        try:
            load_more = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, 'button[data-testid="taxonomy-page__load-more"]')
                )
            )

            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});",
                load_more
            )
            driver.execute_script("arguments[0].click();", load_more)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n[{tag}] Stopped early: {e}")
            break
            
    return driver

def extract_article_content_cointelegraph(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return None, None, f"❌ Request failed: {e}"

    soup = BeautifulSoup(response.content, 'html.parser')

    date_tag = soup.find('span', attrs={'data-testid': 'post-article-meta__publish-date'})
    dt_utc = pd.to_datetime(date_tag.find_all('span')[1].get_text(strip=True)).tz_localize(None) if date_tag else None

    title_tag = soup.find('h1')
    title     = title_tag.get_text(strip=True) if title_tag else None

    article_body = soup.find('div', attrs={'data-testid': 'post__body'})
    if not article_body:
        return dt_utc, title, "❌ Content not found."

    paragraphs = []
    for p in article_body.find_all('p', recursive=True):
        if p.find_parent(['div', 'aside'], class_=lambda c: c and any(
            x in c for x in ['ad', 'embed', 'promo', 'widget', 'related']
        )):
            continue
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text)

    return dt_utc, title, "\n\n".join(paragraphs)

def scrape_cointelegraph(
    tags, save_path,
    num_clicks=50,
    time_sleep_open=3
):
    all_links = {}   # url → tag, deduplicates automatically

    # ── Collect links across all tags ────────────────────────────────────────
    for tag in tags:
        print(f"\nCollecting links for tag: [{tag}]")
        driver = set_up_driver_cointelegraph(tag, num_clicks, time_sleep_open)
        soup   = BeautifulSoup(driver.page_source, 'html.parser')
        driver.quit()

        for a_tag in soup.select('article[data-testid="article-card"] a[data-title-link]'):
            href = a_tag.get('href')
            if not href:
                continue
            if href.startswith('/'):
                href = "https://cointelegraph.com" + href
            if href not in all_links:
                all_links[href] = tag   # first tag wins if duplicated

    print(f"\nTotal unique articles found: {len(all_links)}")

    # ── Extract content from each article ────────────────────────────────────
    records = []
    for url, tag in tqdm(all_links.items(), desc="Extracting articles", unit='article'):
        dt, title, content = extract_article_content_cointelegraph(url)
        records.append({
            'TIME_POSTED': dt,
            'TITLE':       title,
            'CONTENT':     content,
            'TAG':         tag,
            'URL':         url
        })
        time.sleep(0.5)

    # ── Save ─────────────────────────────────────────────────────────────────
    news_df = (
        pd.DataFrame(records)
        .dropna(subset=['TIME_POSTED'])
        .sort_values('TIME_POSTED')
        .reset_index(drop=True)
    )

    with open(save_path, 'wb') as f:
        pickle.dump(news_df, f)

    print(f"\nSaved {len(news_df)} articles → {save_path}")
    print(news_df.groupby('TAG').size().rename('count'))
    return news_df

# COINTELEGRAPH_TAGS = [
#         'markets',
#         'economics',
#         'business',
#         'regulation',
#         'finance',
#         'technology',
#     ]


# For Coindesk and APNews

def _make_session():
    """Requests session with connection pooling + automatic retry."""
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=100,
        pool_maxsize=100,
        max_retries=Retry(total=3, backoff_factor=0.3)
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    return session
 
def _make_driver():
    """
    Headless Chrome with:
      - images / fonts / CSS disabled
      - background services disabled
      - eager page-load strategy (waits for DOM only, not full assets)
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-logging")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-sync")
    options.add_argument("--disable-default-apps")
    options.add_argument("--mute-audio")
    options.add_argument("--user-agent=Mozilla/5.0")
    options.add_argument("--blink-settings=imagesEnabled=false")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2
    })
    options.page_load_strategy = "eager"
 
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
        driver = webdriver.Chrome(options=options)
    return driver
 
def _ckpt_save(path, obj):
    """Atomic-ish pickle save: write to .tmp then rename."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)
 
def _ckpt_load(path):
    """Load pickle checkpoint; return None if file does not exist."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None
 
# ══════════════════════════════════════════════════════════════════════════════
#  CoinDesk — Stage 1: collect links
# ══════════════════════════════════════════════════════════════════════════════
 
def _collect_links_coindesk(args):
    """
    Per-tag worker (separate process).
    Optimised for very large click counts:
    - JS-based counting
    - bulk href extraction
    - periodic DOM pruning to prevent slowdown
    """

    tag, num_clicks, time_sleep_open = args

    # How often to harvest + prune DOM
    harvest_every = 100

    # ──────────────────────────────────────────────────────────────
    # JS snippets
    # ──────────────────────────────────────────────────────────────

    COUNT_JS = """
        return document.querySelectorAll('a.content-card-title').length;
    """
    # Collect hrefs + REMOVE harvested cards from DOM
    HARVEST_AND_PRUNE_JS = """
        const anchors = Array.from(
            document.querySelectorAll('a.content-card-title')
        );

        const hrefs = anchors
            .map(el => el.href)
            .filter(h => h);

        // Remove only the nearest card/article container
        for (const el of anchors) {

            const card =
                el.closest('article') ||
                el.closest('[class*="card"]') ||
                el.closest('[class*="Card"]');

            if (card && card.remove) {
                card.remove();
            }
        }

        return hrefs;
    """

    FINAL_HREFS_JS = """
        return Array.from(
            document.querySelectorAll('a.content-card-title')
        )
        .map(el => el.href)
        .filter(h => h);
    """

    # ──────────────────────────────────────────────────────────────

    driver = _make_driver()

    driver.get(f'https://www.coindesk.com/{tag}')

    WebDriverWait(driver, time_sleep_open + 10).until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "a.content-card-title")
        )
    )

    collected_links = set()

    for i in tqdm(
        range(num_clicks),
        desc=f"[{tag}] Loading pages",
        unit="page"
    ):

        try:
            old_count = driver.execute_script(COUNT_JS)

            load_more = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(., 'More stories')]")
                )
            )

            driver.execute_script(
                "arguments[0].click();",
                load_more
            )

            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script(COUNT_JS) > old_count
            )

            # ──────────────────────────────────────────────
            # Periodic harvest + DOM pruning
            # ──────────────────────────────────────────────

            if (i + 1) % harvest_every == 0:

                harvested = driver.execute_script(
                    HARVEST_AND_PRUNE_JS
                )

                for h in harvested:
                    if h:
                        if h.startswith('/'):
                            h = 'https://www.coindesk.com' + h

                        collected_links.add(h)

                print(
                    f"[{tag}] Harvested/pruned at click "
                    f"{i + 1} | total={len(collected_links)}"
                )

        except Exception as e:
            print(f"\n[{tag}] Stopped early: {e}")
            break

    # ──────────────────────────────────────────────────────────────
    # Final remaining live DOM extraction
    # ──────────────────────────────────────────────────────────────

    remaining = driver.execute_script(FINAL_HREFS_JS)

    for h in remaining:
        if h:
            if h.startswith('/'):
                h = 'https://www.coindesk.com' + h

            collected_links.add(h)

    driver.quit()

    return tag, list(collected_links)
 
def collect_links_coindesk(tags, links_path, num_clicks=100, time_sleep_open=1):
    """
    Stage 1 — collect article links for every tag and save to links_path.
 
    Checkpoint behaviour
    --------------------
    - links_path is both the checkpoint and the final output (same file).
    - After EACH tag finishes its process, the cumulative all_links dict is
      saved immediately.  If the run crashes mid-way, the next call resumes
      automatically: already-finished tags are skipped, only the remaining
      tags are re-scraped.
 
    Parameters
    ----------
    tags        : list[str]  e.g. ['livewire', 'markets', ...]
    links_path  : str        path for the links pickle  (.pkl)
    num_clicks  : int        max 'More stories' clicks per tag
    time_sleep_open : float  extra seconds to wait for first page load
    """
 
    # ── Resume: load whatever tags already finished ────────────────────────────
    all_links: dict = _ckpt_load(links_path) or {}   # {url: tag}
    done_tags  = set(all_links.values())
    todo_tags  = [t for t in tags if t not in done_tags]
 
    if not todo_tags:
        print(f"[Stage 1] All tags already collected ({len(all_links)} links). Nothing to do.")
        return all_links
 
    if done_tags:
        print(f"[Stage 1] Resuming — skipping finished tags: {sorted(done_tags)}")
        print(f"[Stage 1] Remaining tags: {todo_tags}")
 
    # ── Run one Chrome process per remaining tag ───────────────────────────────
    worker_args = [(tag, num_clicks, time_sleep_open) for tag in todo_tags]
 
    with ProcessPoolExecutor(max_workers=min(4, len(todo_tags))) as executor:
        future_to_tag = {
            executor.submit(_collect_links_coindesk, arg): arg[0]
            for arg in worker_args
        }
        for future in as_completed(future_to_tag):
            tag, links = future.result()
            new = 0
            for href in links:
                if href not in all_links:
                    all_links[href] = tag
                    new += 1
 
            # ── Checkpoint immediately after this tag ──────────────────────────
            _ckpt_save(links_path, all_links)
            print(f"[Checkpoint] [{tag}] +{new} new links → {len(all_links)} total  (saved)")
 
    print(f"\n[Stage 1] Done. {len(all_links)} unique links saved → {links_path}")
    return all_links
 
# ══════════════════════════════════════════════════════════════════════════════
#  CoinDesk — Stage 2: extract articles
# ══════════════════════════════════════════════════════════════════════════════

def _extract_coindesk(args):
    url, session = args
    try:
        response = session.get(url, timeout=10)
        if response.status_code != 200:
            return url, None, None, f'HTTP {response.status_code}', None
        soup = BeautifulSoup(response.text, 'lxml')
    except Exception as e:
        return url, None, None, f'Error: {e}', None

    # ── Title ──────────────────────────────────────────────────────────────────
    h1 = soup.find('h1')
    title = h1.get_text(strip=True) if h1 else None

    # ── Datetime: 3-layer fallback ─────────────────────────────────────────────
    dt_utc = None

    # Layer 1: JSON-LD (most reliable on modern CoinDesk)
    if dt_utc is None:
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string or '')
                # Handle both single object and @graph list
                items = data if isinstance(data, list) else data.get('@graph', [data])
                for item in items:
                    raw = item.get('datePublished') or item.get('dateModified')
                    if raw:
                        dt_utc = dateutil_parser.parse(raw).astimezone(timezone.utc).replace(tzinfo=None)
                        break
            except Exception:
                pass
            if dt_utc:
                break

    # Layer 2: <time datetime="..."> tag
    if dt_utc is None:
        time_tag = soup.find('time', attrs={'datetime': True})
        if time_tag:
            try:
                dt_utc = dateutil_parser.parse(time_tag['datetime']).astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                pass

    # Layer 3: original span text heuristic (legacy pages)
    if dt_utc is None:
        for span in soup.find_all('span'):
            text = span.get_text(' ', strip=True)
            if '202' in text and ('a.m.' in text or 'p.m.' in text):
                try:
                    dt_utc = dateutil_parser.parse(
                        text.replace('a.m.', 'AM').replace('p.m.', 'PM')
                    ).astimezone(timezone.utc).replace(tzinfo=None)
                    break
                except Exception:
                    pass

    # ── Content: selector waterfall ────────────────────────────────────────────
    paragraphs = []
    selector_used = None

    body = (
        soup.find('div', attrs={'data-module-name': 'article-body'})  # legacy
        or soup.select_one('div.article-body')
        or soup.select_one('div[class*="article-body"]')
        or soup.select_one('div[class*="ArticleBody"]')
        or soup.select_one('article')
        or soup.select_one('main')
    )

    if body:
        selector_used = (
            body.get('data-module-name')
            or body.get('class', ['unknown'])[0]
        )
        for p in body.find_all('p'):
            text = p.get_text(strip=True)
            if len(text) > 30:
                paragraphs.append(text)

    content = '\n\n'.join(paragraphs) if paragraphs else None
    return url, dt_utc, title, content, selector_used

def extract_articles_coindesk(links_path, save_path, batch_size=1000):
    all_links: dict = _ckpt_load(links_path)
    if not all_links:
        raise FileNotFoundError(f"Links file not found: {links_path}")

    records_ckpt_path = save_path.replace('.pkl', '_ckpt.pkl')
    records: list = _ckpt_load(records_ckpt_path) or []
    done_urls = {r['URL'] for r in records}

    remaining = [(url, tag) for url, tag in all_links.items() if url not in done_urls]
    print(f"[Stage 2] {len(done_urls)} already extracted, {len(remaining)} remaining")

    session = _make_session()
    thread_args = [(url, session) for url, _ in remaining]

    with ThreadPoolExecutor(max_workers=20) as executor:
        for batch_start in range(0, len(thread_args), batch_size):
            batch = thread_args[batch_start: batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(thread_args) + batch_size - 1) // batch_size

            results = list(tqdm(
                executor.map(_extract_coindesk, batch),
                total=len(batch),
                desc=f"[Batch {batch_num}/{total_batches}] Extracting CoinDesk",
                unit="article"
            ))

            for url, dt, title, content, selector in results:
                records.append({
                    'TIME_POSTED':   dt,
                    'TITLE':         title,
                    'CONTENT':       content,
                    'TAG':           all_links[url],
                    'URL':           url,
                    'SELECTOR_USED': selector,      # useful for debugging
                })

            _ckpt_save(records_ckpt_path, records)
            print(f"[Checkpoint] Batch {batch_num}/{total_batches} — {len(records)} records")

    news_df = (
        pd.DataFrame(records)
        .sort_values('TIME_POSTED')
        .reset_index(drop=True)
    )

    # ── Diagnostic breakdown BEFORE any filtering ──────────────────────────────
    print("\n── Extraction audit ──────────────────────────────────────────")
    print(f"  Total records          : {len(news_df)}")
    print(f"  Missing TIME_POSTED    : {news_df['TIME_POSTED'].isna().sum()}")
    print(f"  Missing TITLE          : {news_df['TITLE'].isna().sum()}")
    print(f"  Missing CONTENT        : {news_df['CONTENT'].isna().sum()}")
    print(f"  Fully empty (all 3)    : {news_df[['TIME_POSTED','TITLE','CONTENT']].isna().all(axis=1).sum()}")
    print(f"\n  Selector breakdown:")
    print(news_df['SELECTOR_USED'].value_counts(dropna=False).to_string())

    # ── Save full version (nothing dropped) ───────────────────────────────────
    _ckpt_save(save_path, news_df)

    # ── Clean version: only drop rows with NO content AND NO title ────────────
    clean_df = news_df.dropna(subset=['TITLE', 'CONTENT'], how='all')
    clean_path = save_path.replace('.pkl', '_clean.pkl')
    _ckpt_save(clean_path, clean_df)

    if os.path.exists(records_ckpt_path):
        os.remove(records_ckpt_path)

    print(f"\n[Stage 2] Full : {len(news_df)} articles → {save_path}")
    print(f"[Stage 2] Clean: {len(clean_df)} articles → {clean_path}")
    return clean_df









# ══════════════════════════════════════════════════════════════════════════════
#  AP News — Stage 1: collect links
# ══════════════════════════════════════════════════════════════════════════════

def _collect_links_apnews(args):
    """
    Per-category worker (separate process).

    Optimised for large pagination runs:
    - JS bulk extraction
    - periodic harvest
    - DOM pruning
    - reduced Selenium/WebDriver overhead
    """

    category, num_clicks, time_sleep_open = args

    # Even though AP replaces content more than CoinDesk,
    # pagination pages can still accumulate large DOM trees.
    harvest_every = 100

    # ──────────────────────────────────────────────────────────────
    # JS snippets
    # ──────────────────────────────────────────────────────────────

    HARVEST_AND_PRUNE_JS = """
        const anchors = Array.from(
            document.querySelectorAll('a[href]')
        ).filter(el =>
            el.href &&
            el.href.includes('/article/')
        );

        const hrefs = anchors.map(el => el.href);

        // Remove ONLY nearest article/card containers
        for (const el of anchors) {

            const card =
                el.closest('article') ||
                el.closest('[class*="PagePromo"]') ||
                el.closest('[class*="Card"]') ||
                el.closest('[class*="FeedCard"]');

            if (card && card.remove) {
                card.remove();
            }
        }

        return hrefs;
    """

    FINAL_HREFS_JS = """
        return Array.from(
            document.querySelectorAll('a[href]')
        )
        .map(el => el.href)
        .filter(h => h && h.includes('/article/'));
    """

    # ──────────────────────────────────────────────────────────────

    driver = _make_driver()
    driver.get(f"https://apnews.com/{category}")

    try:
        WebDriverWait(driver, time_sleep_open + 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.PageList-nextPage'))
        )
    except Exception as e:
        print(f"[AP:{category}] Initial page load failed: {e}")
        driver.quit()
        return category, []      # return empty — don't crash siblings

    collected_links = set()

    for i in tqdm(
        range(num_clicks),
        desc=f"[AP:{category}] Loading",
        unit='page'
    ):
        try:
            old_button = WebDriverWait(driver, 15).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, 'div.PageList-nextPage a.Button')
                )
            )
            driver.execute_script(
                "arguments[0].scrollIntoView({block:'center'});",
                old_button
            )
            try:
                old_button.click()
            except Exception:
                driver.execute_script(
                    "arguments[0].click();",
                    old_button
                )
            # Wait until previous button becomes stale
            WebDriverWait(driver, 15).until(
                EC.staleness_of(old_button)
            )
            # Wait until next page controls appear
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'div.PageList-nextPage')
                )
            )
            # ──────────────────────────────────────────────
            # Periodic harvest + DOM pruning
            # ──────────────────────────────────────────────
            if (i + 1) % harvest_every == 0:
                harvested = driver.execute_script(
                    HARVEST_AND_PRUNE_JS
                )
                for h in harvested:
                    if h:
                        if h.startswith('/'):
                            h = 'https://apnews.com' + h
                        collected_links.add(h)
                print(
                    f"[AP:{category}] Harvested/pruned "
                    f"at click {i + 1} | "
                    f"total={len(collected_links)}"
                )
        except Exception as e:

            print(
                f"\n[AP:{category}] "
                f"Stopped at click {i + 1}: {e}"
            )

            break

    # ──────────────────────────────────────────────────────────────
    # Final extraction from remaining live DOM
    # ──────────────────────────────────────────────────────────────

    remaining = driver.execute_script(FINAL_HREFS_JS)
    for h in remaining:
        if h:
            if h.startswith('/'):
                h = 'https://apnews.com' + h
            collected_links.add(h)
    driver.quit()

    return category, list(collected_links)

def collect_links_apnews(categories, links_path, num_clicks=100, time_sleep_open=3):

    ckpt      = _ckpt_load(links_path) or {}
    all_links = ckpt.get('links', {})
    done_cats = ckpt.get('done_cats', set())
    todo_cats = [c for c in categories if c not in done_cats]

    if not todo_cats:
        print(f"[Stage 1] All categories already collected ({len(all_links)} links). Nothing to do.")
        return all_links

    if done_cats:
        print(f"[Stage 1] Resuming — skipping: {sorted(done_cats)}")
        print(f"[Stage 1] Remaining: {todo_cats}")

    worker_args = [(cat, num_clicks, time_sleep_open) for cat in todo_cats]

    with ProcessPoolExecutor(max_workers=min(2, len(todo_cats))) as executor:
        future_to_cat = {
            executor.submit(_collect_links_apnews, arg): arg[0]
            for arg in worker_args
        }
        for future in as_completed(future_to_cat):
            cat = future_to_cat[future]
            try:
                category, links = future.result()
            except Exception as e:
                print(f"[AP:{cat}] Worker crashed: {e} — skipping, will retry on next run")
                continue                             # other categories keep saving

            new = 0
            for href in links:
                if href not in all_links:
                    all_links[href] = category
                    new += 1
            done_cats.add(category)
            _ckpt_save(links_path, {'links': all_links, 'done_cats': done_cats})
            print(f"[Checkpoint] [AP:{category}] +{new} new → {len(all_links)} total")

    print(f"\n[Stage 1] Done. {len(all_links)} links → {links_path}")
    return all_links


# ══════════════════════════════════════════════════════════════════════════════
#  AP News — Stage 2: extract articles
# ══════════════════════════════════════════════════════════════════════════════
 
def _extract_apnews(args):
    """Per-URL worker (thread). Returns (url, dt, title, content)."""
    url, session, delay = args
    time.sleep(delay)
    try:
        response = session.get(url, timeout=15)
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 30))
            time.sleep(retry_after)
            response = session.get(url, timeout=15)
        if response.status_code != 200:
            return url, None, None, f'HTTP {response.status_code}'
    except Exception as e:
        return url, None, None, f'Error: {e}'
 
    soup = BeautifulSoup(response.content, 'lxml')
 
    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else None
 
    dt_utc = None
    time_tag = soup.find('bsp-timestamp')
    if time_tag and time_tag.get('data-timestamp'):
        try:
            ts = time_tag.get('data-timestamp')
            if ts:
                dt_utc = pd.to_datetime(int(ts), unit='ms', utc=True).tz_convert(None)
        except Exception:
            pass
 
    paragraphs = []
    article_body = (
        soup.find('div', class_=lambda x: x and 'RichTextStoryBody' in x)
        or soup.find('div', class_=lambda x: x and 'Article' in (x or ''))
        or soup.select_one('article')
        or soup.select_one('main')
    )
    selector_used = type(article_body).__name__ if article_body else None

    if article_body:
        for p in article_body.find_all('p'):
            text = p.get_text(strip=True)
            if text:
                paragraphs.append(text)
 
    return url, dt_utc, title, "\n\n".join(paragraphs)

def extract_articles_apnews(links_path, save_path, batch_size=1000):

    ckpt = _ckpt_load(links_path)
    if not ckpt:
        raise FileNotFoundError(f"Links file not found: {links_path}")

    # Handle both old format {url:cat} and new format {'links':{url:cat}, ...}
    all_links = ckpt.get('links', ckpt) if isinstance(ckpt, dict) else ckpt

    records_ckpt_path = save_path.replace('.pkl', '_ckpt.pkl')
    records   = _ckpt_load(records_ckpt_path) or []
    done_urls = {r['URL'] for r in records}

    remaining = [(url, cat) for url, cat in all_links.items() if url not in done_urls]
    print(f"[Stage 2] {len(done_urls)} already extracted, {len(remaining)} remaining")

    session     = _make_session()
    thread_args = [
        (url, session, random.uniform(0.5, 2.0))   # ← delay added
        for url, _ in remaining
    ]

    with ThreadPoolExecutor(max_workers=5) as executor:
        for batch_start in range(0, len(thread_args), batch_size):
            batch     = thread_args[batch_start: batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(thread_args) + batch_size - 1) // batch_size

            results = list(tqdm(
                executor.map(_extract_apnews, batch),
                total=len(batch),
                desc=f"[Batch {batch_num}/{total_batches}] Extracting AP News",
                unit="article"
            ))

            for url, dt, title, content in results:
                records.append({
                    'TIME_POSTED': dt,
                    'TITLE':       title,
                    'CONTENT':     content,
                    'CATEGORY':    all_links[url],
                    'URL':         url
                })

            _ckpt_save(records_ckpt_path, records)
            print(f"[Checkpoint] Batch {batch_num}/{total_batches} — {len(records)} records")

    news_df = pd.DataFrame(records)
    _ckpt_save(save_path, news_df)

    if os.path.exists(records_ckpt_path):
        os.remove(records_ckpt_path)

    print(f"\n── Extraction audit ───────────────────────────────────────")
    print(f"  Total records       : {len(news_df)}")
    print(f"  Missing TIME_POSTED : {news_df['TIME_POSTED'].isna().sum()}")
    print(f"  Missing TITLE       : {news_df['TITLE'].isna().sum()}")
    print(f"  Missing CONTENT     : {(news_df['CONTENT'] == '').sum()}")
    print(f"\n[Stage 2] Done. {len(news_df)} articles → {save_path}")
    return news_df
