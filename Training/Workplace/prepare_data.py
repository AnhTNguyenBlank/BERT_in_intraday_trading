
import sys
sys.path.insert(0, 'D:/BERT_in_intraday_trading')
from src.support import *

import pickle
from collections import Counter


import requests
import duckdb
import zipfile
import time
import gc
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


if __name__ == '__main__':

    #=========================== Scrape source for news_data ===========================#

    # driver = set_up_driver(num_clicks = 10000, time_sleep_open = 5, time_sleep_btw = 1)

    # # After loading all articles, parse page source with BeautifulSoup
    # soup = BeautifulSoup(driver.page_source, 'html.parser')
    # driver.quit()

    # # Save to file
    # with open('D:/BERT_in_intraday_trading/Training/Saved_results/page_source_2.html', 'w', encoding='utf-8') as f:
    #     f.write(str(soup))

    # #=========================== Extract news_data from source ===========================#

    # # Reload source from file
    # with open('D:/BERT_in_intraday_trading/Training/Saved_results/page_source_2.html', 'r', encoding='utf-8') as f:
    #     html = f.read()
    # soup = BeautifulSoup(html, 'html.parser')

    # # Reload stored news data
    # with open("D:/BERT_in_intraday_trading/Training/Data/stored_data_2.pkl", "rb") as f:
    #     stored_data = pickle.load(f)

    # # Pre-filter relevant links first
    # valid_links = [
    #     link for link in soup.find_all('a')
    #     if link.string and len(link.string.strip()) > 35 and link.has_attr('href')
    # ]

    # for link in tqdm(valid_links[30000:], desc="Extracting articles' contents", unit = 'article'):
    #     try:
    #         time.sleep(1)
    #         full_url = link['href']
    #         time_posted, content = extract_article_content(full_url)
    #         date_posted = link['href'][-10:]
    #         title = link.string.strip()

    #         temp = dict()

    #         temp['TITLE'] = title
    #         temp['URL'] = full_url
    #         temp['DATE_POSTED'] = date_posted
    #         temp['TIME_POSTED'] = time_posted
    #         temp['CONTENT'] = content

    #         stored_data.append(temp)

    #     except (TooManyRedirects, SSLError, ReadTimeout):
    #         time.sleep(10)
    #         continue

    # with open("D:/BERT_in_intraday_trading/Training/Data/stored_data_2.pkl", "wb") as f:
    #     pickle.dump(stored_data, f)

    
    # #=========================== Load news_data ===========================#

    # with open("D:/BERT_in_intraday_trading/Training/Data/stored_data_2.pkl", "rb") as f:
    #     news_data = pickle.load(f)
        
    # news_data = [new for new in news_data if new['CONTENT'] != 'content']
    # news_data = pd.DataFrame(news_data)
    # news_data.set_index(keys = 'TIME_POSTED', inplace = True)
    # news_data.index = pd.to_datetime(news_data.index).tz_localize(None)
    # news_data = news_data[~news_data.index.isna()].sort_index()

    # COINDESK_SECTIONS = [
    #     'livewire',
    #     'markets',
    #     'business',
    #     'policy',
    #     'tech'
    # ]
 
    # COINDESK_LINKS_PATH   = 'D:/BERT_in_intraday_trading/Training/Data/coindesk_links.pkl'
    # COINDESK_SAVE_PATH    = 'D:/BERT_in_intraday_trading/Training/Data/news_coindesk.pkl'

    # --- Stage 1: collect links (run this first, comment out when done) ---
    # collect_links_coindesk(
    #     tags=COINDESK_SECTIONS,
    #     links_path=COINDESK_LINKS_PATH,
    #     num_clicks=150,
    #     time_sleep_open=1
    # )
 
    # --- Stage 2: extract articles (run after Stage 1 finishes) ---
    # news_df = extract_articles_coindesk(
    #     links_path=COINDESK_LINKS_PATH,
    #     save_path=COINDESK_SAVE_PATH,
    #     batch_size=1000
    # )
    # print(news_df.drop_duplicates(subset = 'TITLE').shape)
 
    # ========================== AP News ========================== #
 
    # AP_NEWS_HUBS = [
    #     'business',
    #     'politics',
    #     'technology',
    #     'economy',
    #     'world-news',
    #     'science',
    # ]
 
    # APNEWS_LINKS_PATH  = 'D:/BERT_in_intraday_trading/Training/Data/apnews_links.pkl'
    # APNEWS_SAVE_PATH   = 'D:/BERT_in_intraday_trading/Training/Data/news_apnews.pkl'
 
    # # --- Stage 1: collect links (run this first, comment out when done) ---
    # # collect_links_apnews(
    # #     categories=AP_NEWS_HUBS,
    # #     links_path=APNEWS_LINKS_PATH,
    # #     num_clicks=30000,
    # #     time_sleep_open=5
    # # )
 
    # # --- Stage 2: extract articles (run after Stage 1 finishes) ---
    # news_df = extract_articles_apnews(
    #     links_path=APNEWS_LINKS_PATH,
    #     save_path=APNEWS_SAVE_PATH,
    #     batch_size=1000
    # )
    # print(news_df)
    
    # news_df = pd.read_pickle(APNEWS_SAVE_PATH)




    # -----------------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------------
    START_DT     = datetime(2025, 5, 31, 0, 0, 0)
    END_DT       = datetime(2026, 5, 1, 23, 45, 0)
    STEP         = timedelta(minutes=15)
    RECONNECT_N  = 100

    DB_PATH  = Path("D:/BERT_in_intraday_trading/Training/Data/gdelt_gkg_2.duckdb")
    TMP_ZIP  = Path("tmp_gkg_2.zip")
    TMP_CSV  = Path("tmp_gkg_2.csv")

    BASE_URL = "http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"

    # -----------------------------------------------------------------------------
    # DB helpers
    # -----------------------------------------------------------------------------
    def get_con():
        con = duckdb.connect(str(DB_PATH))
        con.execute("PRAGMA memory_limit='1GB'")
        con.execute("""
            CREATE TABLE IF NOT EXISTS gkg (
                DATE                   VARCHAR,
                DocumentIdentifier     VARCHAR,
                V2Themes               VARCHAR,
                V2Locations            VARCHAR,
                V2Persons              VARCHAR,
                V2Organizations        VARCHAR,
                tone                   DOUBLE,
                positive_score         DOUBLE,
                negative_score         DOUBLE,
                polarity               DOUBLE,
                activity_density       DOUBLE,
                self_reference_density DOUBLE,
                word_count             DOUBLE
            )
        """)
        con.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_doc
            ON gkg (DocumentIdentifier)
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS processed (
                ts VARCHAR PRIMARY KEY
            )
        """)
        return con

    # -----------------------------------------------------------------------------
    # Timestamps
    # -----------------------------------------------------------------------------
    def all_timestamps(start, end, step):
        dt = start
        while dt <= end:
            yield dt.strftime("%Y%m%d%H%M%S")
            dt += step

    con = get_con()

    already_done = set(
        row[0] for row in con.execute("SELECT ts FROM processed").fetchall()
    )

    timestamps = list(all_timestamps(START_DT, END_DT, STEP))
    remaining  = [ts for ts in timestamps if ts not in already_done]

    print(f"Total timestamps : {len(timestamps)}")
    print(f"Already processed: {len(already_done)}")
    print(f"Remaining        : {len(remaining)}")

    # -----------------------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------------------
    QUERY = r"""
        SELECT
            column01 AS DATE,
            column04 AS DocumentIdentifier,
            column08 AS V2Themes,
            column10 AS V2Locations,
            column12 AS V2Persons,
            column14 AS V2Organizations,

            TRY_CAST(SPLIT(column15, ',')[1] AS DOUBLE) AS tone,
            TRY_CAST(SPLIT(column15, ',')[2] AS DOUBLE) AS positive_score,
            TRY_CAST(SPLIT(column15, ',')[3] AS DOUBLE) AS negative_score,
            TRY_CAST(SPLIT(column15, ',')[4] AS DOUBLE) AS polarity,
            TRY_CAST(SPLIT(column15, ',')[5] AS DOUBLE) AS activity_density,
            TRY_CAST(SPLIT(column15, ',')[6] AS DOUBLE) AS self_reference_density,
            TRY_CAST(SPLIT(column15, ',')[7] AS DOUBLE) AS word_count

        FROM read_csv(
            'tmp_gkg_2.csv',
            delim         = '\t',
            header        = False,
            ignore_errors = true,
            strict_mode   = false,
            null_padding  = true,
            quote         = chr(0),
            all_varchar   = true
        )
        WHERE column08 IS NOT NULL
        AND (

            -- Tier 1: crypto-native sources — only require at least one crypto theme
            (
                (
                    LOWER(column04) LIKE '%bitcoin%'
                    OR LOWER(column04) LIKE '%crypto%'
                    OR LOWER(column04) LIKE '%blockchain%'
                    OR LOWER(column04) LIKE '%coindesk%'
                    OR LOWER(column04) LIKE '%cointelegraph%'
                    OR LOWER(column04) LIKE '%decrypt.co%'
                )
                AND (
                    column08 LIKE '%BITCOIN%'
                    OR column08 LIKE '%CRYPTO%'
                    OR column08 LIKE '%BLOCKCHAIN%'
                    OR column08 LIKE '%DIGITAL_CURRENCY%'
                    OR column08 LIKE '%COIN%'
                    OR column08 LIKE '%FINTECH%'
                )
            )

            OR

            -- Tier 2: general financial sources — must have crypto theme AND economic context
            (
                (
                    LOWER(column04) LIKE '%reuters%'
                    OR LOWER(column04) LIKE '%bloomberg%'
                    OR LOWER(column04) LIKE '%ft.com%'
                    OR LOWER(column04) LIKE '%cnbc%'
                )
                AND (
                    column08 LIKE '%BITCOIN%'
                    OR column08 LIKE '%CRYPTO%'
                    OR column08 LIKE '%BLOCKCHAIN%'
                    OR column08 LIKE '%DIGITAL_CURRENCY%'
                    OR column08 LIKE '%FINTECH%'
                )
                AND (
                    column08 LIKE '%FEDERAL_RESERVE%'
                    OR column08 LIKE '%INTEREST_RATE%'
                    OR column08 LIKE '%REGULATION%'
                    OR column08 LIKE '%SEC%'
                    OR column08 LIKE '%CENTRAL_BANK%'
                    OR column08 LIKE '%BANKING%'
                    OR column08 LIKE '%RECESSION%'
                    OR column08 LIKE '%INFLATION%'
                    OR column08 LIKE '%SANCTION%'
                )
            )
        )
    """

    # -----------------------------------------------------------------------------
    # Process
    # -----------------------------------------------------------------------------
    errors = []

    for i, ts in enumerate(tqdm(remaining, desc="Processing GKG files")):
        url = BASE_URL.format(ts=ts)

        try:
            r = requests.get(url, stream=True, verify=False, timeout=30)
            if r.status_code == 404:
                con.execute("INSERT OR IGNORE INTO processed VALUES (?)", [ts])
                continue
            r.raise_for_status()

            with open(TMP_ZIP, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

            with zipfile.ZipFile(TMP_ZIP, "r") as z:
                names = z.namelist()
                z.extract(names[0], ".")
                Path(names[0]).rename(TMP_CSV)

            con.execute(f"INSERT OR IGNORE INTO gkg {QUERY}")
            con.execute("INSERT OR IGNORE INTO processed VALUES (?)", [ts])

        except Exception as e:
            errors.append((ts, str(e)))
            tqdm.write(f"  SKIP {ts}: {e}")
            time.sleep(1)

        finally:
            TMP_ZIP.unlink(missing_ok=True)
            TMP_CSV.unlink(missing_ok=True)

        if (i + 1) % RECONNECT_N == 0:
            con.execute("CHECKPOINT")
            con.close()
            gc.collect()
            con = get_con()
            tqdm.write(f"  Reconnected at {i + 1} files")

    # Final flush
    con.execute("CHECKPOINT")

    # -----------------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------------
    total_rows = con.execute("SELECT COUNT(*) FROM gkg").fetchone()[0]
    print(f"\nDone. Rows in DB : {total_rows:,}")
    print(f"Errors           : {len(errors)}")
    if errors:
        for ts, msg in errors[:10]:
            print(f"  {ts}: {msg}")

    con.close()