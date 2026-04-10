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

    df_1_min['KEY_MAP_15M'] = df_1_min.index.ceil('15min')
    df_1_min['LOG_RET(T)'] = np.log(df_1_min['CLOSE']/df_1_min['CLOSE'].shift(1))
    news_data['KEY_MAP_15M'] = news_data.index.ceil('15min')

    # Prepare data

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

    df = df.merge(
        df_rv,
        how = 'left',
        left_index = True,
        right_index = True
    )
    df.columns = ['TITLE', 'CONTENT', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TICK_VOL',
       'AVG_PRICE', 'FLAG_INCREASE_CANDLE', 'BODY', 'UPPER_SHADOW',
       'LOWER_SHADOW', 'WHOLE_RANGE', 'FLAG_LONG_UPPER_SHADOW',
       'FLAG_LONG_LOWER_SHADOW', 'FLAG_HIGHER_HIGH(20)', 'FLAG_HIGHER_LOW(20)',
       'AVG_VOL(50)', 'FLAG_OVER_AVG_VOL(50)', 'AVG_VOL(200)',
       'FLAG_OVER_AVG_VOL(200)', 'FLAG_UPTREND_VOL(20)', 'RSI',
       'FLAG_UNDER_30_RSI', 'FLAG_OVER_70_RSI', 'FLAG_UPTREND_RSI(20)',
       'BB_UPPER_BAND(50)', 'POSITION_UPPER_BAND(50)', 'BB_LOWER_BAND(50)',
       'POSITION_LOWER_BAND(50)', 'EMA(50)', 'POSITION_EMA(50)', 'EMA(200)',
       'POSITION_EMA(200)', 'RV(T)'
    ]

    intraday_returns = (
        df_1_min
        .assign(rank = lambda d: d.groupby("KEY_MAP_15M").cumcount())  # 0..14 within each bar
        .pivot(index="KEY_MAP_15M", columns="rank", values="LOG_RET(T)")  # (N, 15)
        .sort_index()
    )

    # Prepare data for training

    data = df[['LOG_RET(T)', 'RV(T)', 'CONTENT']].dropna()
    intraday_returns = intraday_returns[(intraday_returns.index >= data.index[0]) & (intraday_returns.index <= data.index[-1])]
    data = data.merge(intraday_returns, left_index= True, right_index = True, how = 'left').dropna().values
    
    N = len(data)
    num_samples = N - 20

    r_seq = np.zeros((num_samples, 20))
    rv_seq = np.zeros((num_samples, 20))
    r_target = np.zeros((num_samples, 1))  # r_t
    rv_target = np.zeros((num_samples, 1))  # rv_t
    text = np.empty((num_samples, 1), dtype=object)

    for i in range(num_samples):
        r_seq[i, :] = data[i:i+20, 0]      # past 20 returns
        rv_seq[i, :] = data[i:i+20, 1]      # past 20 rv

        r_target[i, 0] = data[i+20, 0]     # next return
        rv_target[i, 0] = data[i+20, 1]     # next rv
        text[i, 0] = data[i+20, 2]     # next return

    r_seq = tf.expand_dims(r_seq, axis=-1)
    rv_seq = tf.expand_dims(rv_seq, axis=-1)
    text = tf.squeeze(text)
    r_target = tf.convert_to_tensor(r_target)
    rv_target = tf.convert_to_tensor(rv_target)

    intraday_returns = data[20:, 3:]

    save_dir = "./Training/Saved_results"
    os.makedirs(save_dir, exist_ok=True)

    # DeepARCH
    deep_arch = DeepARCH(lstm_units=20)
    deep_arch.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    deep_arch(tf.zeros((1, 20, 1)))
    deep_arch.load_weights(os.path.join(save_dir, "deep_arch.weights.h5"))

    # DeepRARCH
    deep_rarch = DeepRARCH(lstm_units=20)
    deep_rarch.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    deep_rarch(tf.zeros((1, 20, 1)), tf.zeros((1, 20, 1)))
    deep_rarch.load_weights(os.path.join(save_dir, "deep_rarch.weights.h5"))

    # DeepLLMRARCH
    deep_llmrarch = DeepLLMRARCH(lstm_units=20)
    deep_llmrarch.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    deep_llmrarch(tf.zeros((1, 20, 1)), tf.zeros((1, 20, 1)), tf.constant(["dummy news text"]))
    deep_llmrarch.load_weights(os.path.join(save_dir, "deep_llmrarch.weights.h5"))

    print("All model weights loaded.")

    # Evaluation

    evaluator = Evaluator(compare_variance = False)

    log_sigma2 = deep_arch.predict(r_seq)
    results_error, results_tail, NLL = evaluator.evaluate(log_sigma2 = log_sigma2,
                                                      returns_target = r_target,
                                                      intraday_returns = intraday_returns,
                                                      alpha_levels = [0.01, 0.05])
    print(results_error)
    print(results_tail)
    print(f'Negative loglikelihood: {NLL}')