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
from arch import arch_model

from src.support import *
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

    # Visualize data

    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (20, 8))
    ax[0].plot(df['TITLE'], alpha = 0.5)

    ax_0 = ax[0].twinx()
    ax_0.plot(df['CLOSE'],
            color = 'red'
            )

    ax[1].boxplot(df['TITLE'])

    plt.tight_layout()
    plt.show()

    # Prepare data for training

    data = df[['LOG_RET(T)', 'RV(T)', 'CONTENT']].dropna()
    intraday_returns = intraday_returns[(intraday_returns.index >= data.index[0]) & (intraday_returns.index <= data.index[-1])]
    data = data.merge(intraday_returns, left_index= True, right_index = True, how = 'left').dropna().values
    
    N = len(data)
    window = 20
    num_samples = N - window

    r_seq = np.zeros((num_samples, window))
    rv_seq = np.zeros((num_samples, window))
    r_target = np.zeros((num_samples, 1))  # r_t
    rv_target = np.zeros((num_samples, 1))  # rv_t
    text = np.empty((num_samples, 1), dtype=object)

    for i in range(num_samples):
        r_seq[i, :] = data[i:i+window, 0]      # past returns
        rv_seq[i, :] = data[i:i+window, 1]      # past rv
        text[i, 0] = data[i+window-1, 2]     # news

        r_target[i, 0] = data[i+window, 0]     # next return
        rv_target[i, 0] = data[i+window, 1]     # next rv
        
    r_seq = tf.expand_dims(r_seq, axis=-1)
    rv_seq = tf.expand_dims(rv_seq, axis=-1)
    text = tf.squeeze(text)
    r_target = tf.convert_to_tensor(r_target)
    rv_target = tf.convert_to_tensor(rv_target)

    intraday_returns = data[window:, 3:]
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

    # GARCH(1,1)
    train_returns = r_target_train[:, 0].numpy()
    test_returns = r_target_test[:, 0].numpy()

    am = arch_model(train_returns)
    res = am.fit(update_freq=5)
    
    save_dir = Path("/content/drive/MyDrive/Projects/BERT_in_intraday_trading/Training/Saved_results")
    save_dir.mkdir(parents=True, exist_ok=True)

    params_dict = {
        k: float(v)
        for k, v in res.params.items()
    }

    # save as jsonl
    jsonl_path = save_dir / "garch_params.jsonl"

    with open(jsonl_path, "w") as f:
        f.write(json.dumps(params_dict) + "\n")

    log_sigma2_GARCH_train = np.log(res.conditional_volatility**2 + 1e-8)

    omega = res.params['omega']
    alpha = res.params['alpha[1]']
    beta = res.params['beta[1]']

    # training conditional volatility
    train_sigma2 = res.conditional_volatility**2

    # initialize with last train variance
    last_sigma2 = train_sigma2[-1]

    test_sigma2 = np.zeros(len(test_returns))

    for t in range(len(test_returns)):

        if t == 0:
            prev_return = train_returns[-1]
            prev_sigma2 = last_sigma2
        else:
            prev_return = test_returns[t-1]
            prev_sigma2 = test_sigma2[t-1]

        test_sigma2[t] = (
            omega
            + alpha * prev_return**2
            + beta * prev_sigma2
        )

    log_sigma2_GARCH_test = np.log(test_sigma2)

    train_pickle_path = save_dir / "log_sigma2_GARCH_train.pkl"
    test_pickle_path  = save_dir / "log_sigma2_GARCH_test.pkl"

    with open(train_pickle_path, "wb") as f:
        pickle.dump(log_sigma2_GARCH_train, f)

    with open(test_pickle_path, "wb") as f:
        pickle.dump(log_sigma2_GARCH_test, f)

    # Deep ARCH

    deep_arch = DeepARCH(lstm_units = 20)
    deep_arch.compile(optimizer = tf.keras.optimizers.Adam(1e-3))
    dummy = tf.zeros((1, 20, 1))   # example
    deep_arch(dummy)

    history = deep_arch.fit(
        r_seq, r_target,
        epochs = 200,
        batch_size = 128
    )    
    
    fig,ax = plt.subplots()
    l1 = ax.plot(history.history['loss'],
                #  color="red",
                label = 'Train loss',
                linewidth = 1.5
                )
    # l2  = ax.plot(history.history['val_loss'],
    #              color="orange",
    #              label = 'Test mse',
    #              linewidth = 1.5
    #              )

    # set x-axis label
    ax.set_xlabel("epochs", fontsize = 14)
    ax.set_ylabel("Log-likelihood", fontsize = 14)
    ax.set_title("DeepARCH", fontsize = 14)

    plt.show()

    log_sigma2_DA_train = deep_arch.predict(r_seq_train)
    log_sigma2_DA_test = deep_arch.predict(r_seq_test)

    save_dir = Path("/content/drive/MyDrive/Projects/BERT_in_intraday_trading/Training/Saved_results")
    save_dir.mkdir(parents=True, exist_ok=True)

    deep_arch.save_weights(os.path.join(save_dir, "deep_arch.weights.h5"))

    with open(save_dir / "log_sigma2_DA_train.pkl", "wb") as f:
        pickle.dump(log_sigma2_DA_train, f)

    with open(save_dir / "log_sigma2_DA_test.pkl", "wb") as f:
        pickle.dump(log_sigma2_DA_test, f)

    # Deep Realized ARCH

    deep_rarch = DeepRARCH(lstm_units = 20)
    deep_rarch.compile(optimizer = tf.keras.optimizers.Adam(1e-3))

    dummy = tf.zeros((1, 20, 1))   # example
    deep_rarch((dummy, dummy))

    print(deep_rarch.summary())

    history = deep_rarch.fit(
        x=[r_seq_train, rv_seq_train],
        y=[r_target_train, rv_target_train],
        epochs=200,
        batch_size=128
    )

    fig,ax = plt.subplots()
    l1 = ax.plot(history.history['loss'],
                #  color="red",
                label = 'Train loss',
                linewidth = 1.5
                )
    # l2  = ax.plot(history.history['val_loss'],
    #              color="orange",
    #              label = 'Test mse',
    #              linewidth = 1.5
    #              )

    # set x-axis label
    ax.set_xlabel("epochs", fontsize = 14)
    ax.set_ylabel("Log-likelihood", fontsize = 14)
    ax.set_title("DeepRARCH", fontsize = 14)

    plt.show()

    pred_log_sigma2, pred_rv_hat, pred_log_sigmau2 = deep_rarch.predict((r_seq_train, rv_seq_train))

    log_sigma2_DRA_train = pred_log_sigma2.squeeze()
    rv_hat_DRA_train = pred_rv_hat.squeeze()
    log_sigmau2_DRA_train = pred_log_sigmau2[0]

    pred_log_sigma2, pred_rv_hat, pred_log_sigmau2 = deep_rarch.predict((r_seq_test, rv_seq_test))

    log_sigma2_DRA_test = pred_log_sigma2.squeeze()
    rv_hat_DRA_test = pred_rv_hat.squeeze()
    log_sigmau2_DRA_test = pred_log_sigmau2[0]

    save_dir = Path("/content/drive/MyDrive/Projects/BERT_in_intraday_trading/Training/Saved_results")
    save_dir.mkdir(parents=True, exist_ok=True)

    deep_rarch.save_weights(os.path.join(save_dir, "deep_rarch.weights.h5"))

    with open(save_dir / "log_sigma2_DRA_train.pkl", "wb") as f:
        pickle.dump(log_sigma2_DRA_train, f)

    with open(save_dir / "log_sigma2_DRA_test.pkl", "wb") as f:
        pickle.dump(log_sigma2_DRA_test, f)

    with open(save_dir / "rv_hat_DRA_train.pkl", "wb") as f:
        pickle.dump(rv_hat_DRA_train, f)

    with open(save_dir / "rv_hat_DRA_test.pkl", "wb") as f:
        pickle.dump(rv_hat_DRA_test, f)

    with open(save_dir / "log_sigmau2_DRA_train.pkl", "wb") as f:
        pickle.dump(log_sigmau2_DRA_train, f)

    with open(save_dir / "log_sigmau2_DRA_test.pkl", "wb") as f:
        pickle.dump(log_sigmau2_DRA_test, f)


    # Deep LLM RARCH
    
    deep_llm_rarch = DeepLLMRARCH(lstm_units = 20)
    deep_llm_rarch.compile(optimizer = tf.keras.optimizers.Adam(1e-3))
    dummy = tf.zeros((1, 20, 1))
    dummy_text = tf.constant(["dummy news text"])
    deep_llm_rarch((dummy, dummy, dummy_text))

    print(deep_llm_rarch.summary())

    history = deep_llm_rarch.fit(
        x=[r_seq_train, rv_seq_train, text_train],
        y=[r_target_train, rv_target_train],
        epochs=5,
        batch_size=128
    )
    fig,ax = plt.subplots()
    l1 = ax.plot(history.history['loss'],
                #  color="red",
                label = 'Train loss',
                linewidth = 1.5
                )
    # l2  = ax.plot(history.history['val_loss'],
    #              color="orange",
    #              label = 'Test mse',
    #              linewidth = 1.5
    #              )

    # set x-axis label
    ax.set_xlabel("epochs", fontsize = 14)
    ax.set_ylabel("Log-likelihood", fontsize = 14)
    ax.set_title("DeepLLMRARCH", fontsize = 14)

    plt.show()

    pred_log_sigma2, pred_rv_hat, pred_log_sigmau2 = deep_llm_rarch.predict((r_seq_train, rv_seq_train, text_train))

    log_sigma2_DLRA_train = pred_log_sigma2.squeeze()
    rv_hat_DLRA_train = pred_rv_hat.squeeze()
    log_sigmau2_DLRA_train = pred_log_sigmau2[0]

    pred_log_sigma2, pred_rv_hat, pred_log_sigmau2 = deep_llm_rarch.predict((r_seq_test, rv_seq_test, text_test))

    log_sigma2_DLRA_test = pred_log_sigma2.squeeze()
    rv_hat_DLRA_test = pred_rv_hat.squeeze()
    log_sigmau2_DLRA_test = pred_log_sigmau2[0]

    save_dir = Path("/content/drive/MyDrive/Projects/BERT_in_intraday_trading/Training/Saved_results")
    save_dir.mkdir(parents=True, exist_ok=True)

    deep_llm_rarch.save_weights(os.path.join(save_dir, "deep_llm_rarch.weights.h5"))

    with open(save_dir / "log_sigma2_DLRA_train.pkl", "wb") as f:
        pickle.dump(log_sigma2_DLRA_train, f)

    with open(save_dir / "log_sigma2_DLRA_test.pkl", "wb") as f:
        pickle.dump(log_sigma2_DLRA_test, f)

    with open(save_dir / "rv_hat_DLRA_train.pkl", "wb") as f:
        pickle.dump(rv_hat_DLRA_train, f)

    with open(save_dir / "rv_hat_DLRA_test.pkl", "wb") as f:
        pickle.dump(rv_hat_DLRA_test, f)

    with open(save_dir / "log_sigmau2_DLRA_train.pkl", "wb") as f:
        pickle.dump(log_sigmau2_DLRA_train, f)

    with open(save_dir / "log_sigmau2_DLRA_test.pkl", "wb") as f:
        pickle.dump(log_sigmau2_DLRA_test, f)


    deep_llmrarch.save_weights(os.path.join(save_dir, "deep_llmrarch.weights.h5"))

    print("All model weights saved.")


