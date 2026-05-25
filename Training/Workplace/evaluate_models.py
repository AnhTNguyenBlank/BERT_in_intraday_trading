import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

plt.style.use('classic')
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_rows', 300)

# %config inlinebackend.figure_format = 'svg'

import sys
sys.path.insert(0, 'D:/BERT_in_intraday_trading')

import pickle
import tensorflow as tf

import os

from src.support import *
from src.backtest import Evaluator
from src.models import DeepARCH, DeepRARCH, DeepLLMRARCH

import json
import pickle
from pathlib import Path

if __name__ == '__main__':

    # Import data

    with open(f"./Training/Data/stored_data_2.pkl", "rb") as f:
        news_data = pickle.load(f)

    news_data = [new for new in news_data if new['CONTENT'] != 'content']
    news_data = pd.DataFrame(news_data)
    news_data.set_index(keys = 'TIME_POSTED', inplace = True)
    news_data.index = pd.to_datetime(news_data.index).tz_localize(None)
    news_data = news_data[~news_data.index.isna()].sort_index()

    ohlc_1m = pd.read_pickle(f'./Training/Data/BTCUSD_ohlc_1M.pkl')

    ohlc_1m.columns = [col.upper() for col in ohlc_1m.columns]
    ohlc_1m = ohlc_1m.set_index('TIME')
    ohlc_1m.index = pd.to_datetime(ohlc_1m.index)
    ohlc_1m.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICK_VOL', 'SPREAD', 'REAL_VOLUME', 'FLAG_CANDLE_END']

    df_1_min = prepare_df(df = ohlc_1m, timeframe = '1min', add_indicators = True)
    df_1_min = df_1_min[(df_1_min.index >= news_data.index[0]) & ((df_1_min.index <= news_data.index[-1]))]

    ohlc_15m = pd.read_pickle(f'./Training/Data/BTCUSD_ohlc_15M.pkl')

    ohlc_15m.columns = [col.upper() for col in ohlc_15m.columns]
    ohlc_15m = ohlc_15m.set_index('TIME')
    ohlc_15m.index = pd.to_datetime(ohlc_15m.index)
    ohlc_15m.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICK_VOL', 'SPREAD', 'REAL_VOLUME', 'FLAG_CANDLE_END']

    df_15_min = prepare_df(df = ohlc_15m, timeframe = '15min', add_indicators = True)
    df_15_min = df_15_min[(df_15_min.index >= news_data.index[0]) & ((df_15_min.index <= news_data.index[-1]))]

    df_1_min['KEY_MAP_15M'] = df_1_min.index.floor('15min')
    df_1_min['LOG_RET(T)'] = np.log(df_1_min['CLOSE']/df_1_min['CLOSE'].shift(1))*100
    news_data['KEY_MAP_15M'] = news_data.index.floor('15min')
    # news_data['KEY_MAP_15M'] = np.where(news_data['KEY_MAP_15M'].dt.day_of_week == 6, news_data['KEY_MAP_15M'] - pd.Timedelta(days = 2), news_data['KEY_MAP_15M'])

    df_15_min['LOG_RET(T)'] = np.log(df_15_min['CLOSE']/df_15_min['CLOSE'].shift(1))*100
    
    #===================================== Prepare data
    df_rv = df_1_min.groupby(
        by = 'KEY_MAP_15M'
    ).agg(
        {
            'LOG_RET(T)': lambda x: (x**2).sum()
        }
    )

    df = pd.DataFrame(index = df_15_min[(df_15_min.index >= news_data.index[0]) & ((df_15_min.index <= news_data.index[-1]))].index)
    df = df.merge(
        pd.pivot_table(
            news_data,
            index = 'KEY_MAP_15M',
            values = 'TITLE',
            aggfunc = 'count'
        ),
        how = 'left',
        left_index = True,
        right_index = True
    ).fillna(0)

    df = df.merge(
        news_data.groupby(by = 'KEY_MAP_15M').agg(
            {'TITLE': 'sum'}
        ),
        how = 'left',
        left_index = True,
        right_index = True
    ).fillna('None')

    df = df.merge(
        news_data.groupby(by = 'KEY_MAP_15M').agg(
            {'CONTENT': 'sum'}
        ),
        how = 'left',
        left_index = True,
        right_index = True
    ).fillna('None')

    df = df.merge(
        df_15_min,
        how = 'left',
        left_index = True,
        right_index = True
    )

    df_rv = df_rv.rename(columns={'LOG_RET(T)': 'RV(T)'})
    df = df.merge(
        df_rv,
        how = 'left',
        left_index = True,
        right_index = True
    )

    df.columns = ['NUM_NEWS', 'TITLE', 'CONTENT', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICK_VOL',
       'AVG_PRICE', 'FLAG_INCREASE_CANDLE', 'BODY', 'UPPER_SHADOW',
       'LOWER_SHADOW', 'WHOLE_RANGE', 'FLAG_LONG_UPPER_SHADOW',
       'FLAG_LONG_LOWER_SHADOW', 'FLAG_HIGHER_HIGH(20)', 'FLAG_HIGHER_LOW(20)',
       'AVG_VOL(50)', 'FLAG_OVER_AVG_VOL(50)', 'AVG_VOL(200)',
       'FLAG_OVER_AVG_VOL(200)', 'FLAG_UPTREND_VOL(20)', 'RSI',
       'FLAG_UNDER_30_RSI', 'FLAG_OVER_70_RSI', 'FLAG_UPTREND_RSI(20)',
       'BB_UPPER_BAND(50)', 'POSITION_UPPER_BAND(50)', 'BB_LOWER_BAND(50)',
       'POSITION_LOWER_BAND(50)', 'EMA(50)', 'POSITION_EMA(50)', 'EMA(200)',
       'POSITION_EMA(200)', 'LOG_RET(T)', 'RV(T)']

    intraday_returns = (
        df_1_min
        .assign(rank = lambda d: d.groupby("KEY_MAP_15M").cumcount())  # 0..14 within each bar
        .pivot(index="KEY_MAP_15M", columns="rank", values="LOG_RET(T)")  # (N, 15)
        .sort_index()
    )

    # Visualize data

    fig, ax = plt.subplots(nrows = 4, ncols = 1, figsize = (20, 8))

    ax[0].plot(df['NUM_NEWS'], alpha = 0.5)

    ax_0 = ax[0].twinx()
    ax_0.plot(df['LOG_RET(T)'],
            color = 'red'
            )
    # ax_0.plot(df['RV(T)'],
    #           color = 'green')

    ax[1].boxplot(df['NUM_NEWS'])
    ax[2].hist(df['LOG_RET(T)'], bins = 100)
    ax[3].hist(df['RV(T)'], bins = 100)

    plt.tight_layout()
    plt.show()

    #===================================== Prepare data for training

    data = df[['LOG_RET(T)', 'RV(T)', 'TITLE']].dropna()
    intraday_returns = intraday_returns[(intraday_returns.index >= data.index[0]) & (intraday_returns.index <= data.index[-1])]
    data = data.merge(intraday_returns, left_index= True, right_index = True, how = 'left').dropna().values
    
    N = len(data)
    window = 20
    num_samples = N - window

    content_series = data['TITLE'].values
    numeric_data   = data.drop(columns=['TITLE']).values.astype(np.float64)

    r_seq    = np.zeros((num_samples, window))
    rv_seq   = np.zeros((num_samples, window))
    r_target = np.zeros((num_samples, 1))
    rv_target= np.zeros((num_samples, 1))
    text     = np.empty((num_samples, 1), dtype=object)

    for i in range(num_samples):
        r_seq[i, :]   = numeric_data[i:i+window, 0]
        rv_seq[i, :]  = numeric_data[i:i+window, 1]
        text[i, 0]    = content_series[i+window-1]

        r_target[i, 0]  = numeric_data[i+window, 0]
        rv_target[i, 0] = numeric_data[i+window, 1]

    intraday_returns = numeric_data[window:, 2:]

    train_size = int(num_samples * 0.6)

    # sequences
    r_seq_train = r_seq[:train_size]
    r_seq_test = r_seq[train_size:]

    rv_seq_train = rv_seq[:train_size]
    rv_seq_test = rv_seq[train_size:]

    # targets
    r_target_train = r_target[:train_size]
    r_target_test = r_target[train_size:]

    rv_target_train = rv_target[:train_size]
    rv_target_test = rv_target[train_size:]

    # text
    text_train = text[:train_size]
    text_test = text[train_size:]

    # intraday returns
    intraday_train = intraday_returns[:train_size]
    intraday_test = intraday_returns[train_size:]

    r_seq_train  = tf.expand_dims(r_seq_train,  axis=-1)
    r_seq_test   = tf.expand_dims(r_seq_test,   axis=-1)
    rv_seq_train = tf.expand_dims(rv_seq_train, axis=-1)
    rv_seq_test  = tf.expand_dims(rv_seq_test,  axis=-1)

    r_target_train  = tf.convert_to_tensor(r_target_train,  dtype=tf.float32)
    r_target_test   = tf.convert_to_tensor(r_target_test,   dtype=tf.float32)
    rv_target_train = tf.convert_to_tensor(rv_target_train, dtype=tf.float32)
    rv_target_test  = tf.convert_to_tensor(rv_target_test,  dtype=tf.float32)

    text_train = tf.constant(text_train, dtype=tf.string)
    text_test  = tf.constant(text_test, dtype=tf.string)

    
    # GARCH(1,1)
    save_dir = Path("/content/drive/MyDrive/Projects/BERT_in_intraday_trading/Training/Saved_results")
    jsonl_path = save_dir / "garch_params.jsonl"

    with open(jsonl_path, "r") as f:
        garch_params = json.loads(f.readline())

    print("GARCH Parameters:")
    print(garch_params)

    train_pickle_path = save_dir / "log_sigma2_GARCH_train.pkl"
    with open(train_pickle_path, "rb") as f:
        log_sigma2_GARCH_train = pickle.load(f)

    print("\nTrain log_sigma2 shape:")
    print(log_sigma2_GARCH_train.shape)

    test_pickle_path = save_dir / "log_sigma2_GARCH_test.pkl"
    with open(test_pickle_path, "rb") as f:
        log_sigma2_GARCH_test = pickle.load(f)

    print("\nTest log_sigma2 shape:")
    print(log_sigma2_GARCH_test.shape)

    evaluator = Evaluator(compare_variance = False)
    results_error, results_tail, NLL = evaluator.evaluate(log_sigma2 = log_sigma2_GARCH_train.reshape(-1, 1),
                                                        returns_target = r_target_train,
                                                        intraday_returns = intraday_train,
                                                        alpha_levels = [0.01, 0.05])
    print(results_error)
    print(results_tail)
    print(f'Negative loglikelihood: {NLL}')

    results_error, results_tail, NLL = evaluator.evaluate(log_sigma2 = log_sigma2_GARCH_test,
                                                        returns_target = r_target_test,
                                                        intraday_returns = intraday_test,
                                                        alpha_levels = [0.01, 0.05])
    print(results_error)
    print(results_tail)
    print(f'Negative loglikelihood: {NLL}')

    # DeepARCH
    deep_arch = DeepARCH(lstm_units=20)
    deep_arch.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    deep_arch(tf.zeros((1, 20, 1)))
    deep_arch.load_weights(os.path.join(save_dir, "deep_arch.weights.h5"))

    # Load arrays
    with open(save_dir / "log_sigma2_DA_train.pkl", "rb") as f:
        log_sigma2_DA_train = pickle.load(f)

    with open(save_dir / "log_sigma2_DA_test.pkl", "rb") as f:
        log_sigma2_DA_test = pickle.load(f)

    results_error, results_tail, NLL = evaluator.evaluate(log_sigma2 = log_sigma2_DA_train,
                                                        returns_target = r_target_train,
                                                        intraday_returns = intraday_train,
                                                        alpha_levels = [0.01, 0.05])
    print(results_error)
    print(results_tail)
    print(f'Negative loglikelihood: {NLL}')

    results_error, results_tail, NLL = evaluator.evaluate(log_sigma2 = log_sigma2_DA_test,
                                                        returns_target = r_target_test,
                                                        intraday_returns = intraday_test,
                                                        alpha_levels = [0.01, 0.05])
    print(results_error)
    print(results_tail)
    print(f'Negative loglikelihood: {NLL}')

    # Deep RARCH
    deep_rarch = DeepRARCH(lstm_units=20)
    deep_rarch.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    deep_rarch((tf.zeros((1, 20, 1)), tf.zeros((1, 20, 1))))
    deep_rarch.load_weights(os.path.join(save_dir, "deep_rarch.weights.h5"))

    # Load arrays
    with open(save_dir / "log_sigma2_DRA_train.pkl", "rb") as f:
        log_sigma2_DRA_train = pickle.load(f)

    with open(save_dir / "log_sigma2_DRA_test.pkl", "rb") as f:
        log_sigma2_DRA_test = pickle.load(f)

    with open(save_dir / "rv_hat_DRA_train.pkl", "rb") as f:
        rv_hat_DRA_train = pickle.load(f)

    with open(save_dir / "rv_hat_DRA_test.pkl", "rb") as f:
        rv_hat_DRA_test = pickle.load(f)

    with open(save_dir / "log_sigmau2_DRA_train.pkl", "rb") as f:
        log_sigmau2_DRA_train = pickle.load(f)

    with open(save_dir / "log_sigmau2_DRA_test.pkl", "rb") as f:
        log_sigmau2_DRA_test = pickle.load(f)

    results_error, results_tail, NLL = evaluator.evaluate(
        log_sigma2 = log_sigma2_DRA_train,
        returns_target = r_target,
        intraday_returns = intraday_returns,
        alpha_levels = [0.01, 0.05],
        rv = rv_target,
        rv_hat = rv_hat_DRA_train,
        log_sigmau2 = log_sigmau2_DRA_train
    )

    print(results_error)
    print(results_tail)
    print(f'Negative loglikelihood: {NLL}')

    results_error, results_tail, NLL = evaluator.evaluate(
        log_sigma2 = log_sigma2_DRA_test,
        returns_target = r_target,
        intraday_returns = intraday_returns,
        alpha_levels = [0.01, 0.05],
        rv = rv_target,
        rv_hat = rv_hat_DRA_test,
        log_sigmau2 = log_sigmau2_DRA_test
    )

    print(results_error)
    print(results_tail)
    print(f'Negative loglikelihood: {NLL}')

    # Deep LLM RARCH

    deep_llm_rarch = DeepLLMRARCH(lstm_units=20)
    deep_llm_rarch.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    dummy_x = tf.zeros((1, 20, 1))
    dummy_rv = tf.zeros((1, 20, 1))
    dummy_text = tf.constant(["Example market news text stream."], dtype=tf.string)
    deep_llm_rarch((dummy_x, dummy_rv, dummy_text))
    deep_llm_rarch.load_weights(os.path.join(save_dir, "deep_llm_rarch.weights.h5"))

    # Load arrays
    with open(save_dir / "log_sigma2_DLRA_train.pkl", "rb") as f:
        log_sigma2_DLRA_train = pickle.load(f)

    with open(save_dir / "log_sigma2_DLRA_test.pkl", "rb") as f:
        log_sigma2_DLRA_test = pickle.load(f)

    with open(save_dir / "rv_hat_DLRA_train.pkl", "rb") as f:
        rv_hat_DLRA_train = pickle.load(f)

    with open(save_dir / "rv_hat_DLRA_test.pkl", "rb") as f:
        rv_hat_DLRA_test = pickle.load(f)

    with open(save_dir / "log_sigmau2_DLRA_train.pkl", "rb") as f:
        log_sigmau2_DLRA_train = pickle.load(f)

    with open(save_dir / "log_sigmau2_DLRA_test.pkl", "rb") as f:
        log_sigmau2_DLRA_test = pickle.load(f)

    results_error, results_tail, NLL = evaluator.evaluate(
        log_sigma2 = log_sigma2_DLRA_train,
        returns_target = r_target,
        intraday_returns = intraday_returns,
        alpha_levels = [0.01, 0.05],
        rv = rv_target,
        rv_hat = rv_hat_DLRA_train,
        log_sigmau2 = log_sigmau2_DLRA_train
    )

    print(results_error)
    print(results_tail)
    print(f'Negative loglikelihood: {NLL}')

    results_error, results_tail, NLL = evaluator.evaluate(
        log_sigma2 = log_sigma2_DLRA_test,
        returns_target = r_target,
        intraday_returns = intraday_returns,
        alpha_levels = [0.01, 0.05],
        rv = rv_target,
        rv_hat = rv_hat_DLRA_test,
        log_sigmau2 = log_sigmau2_DLRA_test
    )

    print(results_error)
    print(results_tail)
    print(f'Negative loglikelihood: {NLL}')





