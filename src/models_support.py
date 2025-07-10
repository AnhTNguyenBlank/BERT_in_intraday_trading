import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)
from scipy.stats import levy_stable

from datetime import datetime
from scipy.stats import kstest
from scipy.stats import jarque_bera
# from arch.unitroot import ADF
from scipy.stats import kurtosis
from scipy.stats import skew
# from arch import arch_model

import pickle

import ta

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope

from scipy.signal import argrelmin
from scipy.signal import argrelmax

plt.style.use('classic')


import MetaTrader5 as mt
import pandas as pd
import ml_collections
import yaml

from datetime import datetime
import pytz
import sys

import sys
sys.path.insert(0, 'D:/Intraday_trading')

from src.support import *

#=======================SUPPORT, RESISTANCE=======================

def prepare_df_sr(sr_range, patience_range, patience_time, df_observe, type):

    cnt = 0

    bins = []

    while df_observe['LOW'].min() + cnt*sr_range <= df_observe['HIGH'].max():
        bins.append(df_observe['LOW'].min() + cnt*sr_range)
        cnt += 1

    if type == 'SUPPORT':
        price_type = 'LOW'
    elif type == 'RESISTANCE':
        price_type = 'HIGH'
    elif type == None:
        price_type = 'AVG_PRICE'

    df_observe['PRICE_RANGE'] = pd.cut(df_observe[price_type], bins = bins)

    price_hist = pd.pivot_table(df_observe.copy(), index = 'PRICE_RANGE', values = price_type, aggfunc = 'count')
    price_hist.columns = ['NUM_TOUCH']
    price_hist['PRICE_RANKING'] = range(1, len(price_hist) + 1, 1)
    price_hist['CUMULATIVE_TOUCH'] = price_hist['NUM_TOUCH'].cumsum()

    price_hist['PERC_NUM_TOUCH'] = price_hist['NUM_TOUCH']/price_hist['NUM_TOUCH'].sum()*100
    price_hist['PERC_CUM_NUM_TOUCH'] = price_hist['CUMULATIVE_TOUCH']/price_hist['NUM_TOUCH'].sum()*100

    price_hist = price_hist[['PRICE_RANKING', 'NUM_TOUCH', 'CUMULATIVE_TOUCH', 'PERC_NUM_TOUCH', 'PERC_CUM_NUM_TOUCH']]


    #=================== ADJUST POINTS 1 ===================#
    '''
    Percentile rank based on the number of times prices touch the range;
    +0.5 or -0.5 based on patience_range, i.e. +0.5 to ranks that is maximum within patience_range and -1 otherwise
    --> eliminate the chances that 2 consecutive ranges being support and/or resistance.
    '''
    price_hist['RANK_NUM_TOUCH'] = price_hist['PERC_NUM_TOUCH'].rank(method = 'first', pct = True)

    # Create a forward rolling view
    price_hist['ROLL_RANK_FORWARD'] = price_hist['RANK_NUM_TOUCH'].rolling(window = patience_range).max()

    # Create a backward rolling view by reversing the DataFrame and applying forward rolling
    price_hist['ROLL_RANK_BACKWARD'] = price_hist[::-1]['RANK_NUM_TOUCH'].rolling(window = patience_range).max()
    price_hist['ADJUST_POINTS_1'] = np.where((price_hist['ROLL_RANK_FORWARD'] == price_hist['ROLL_RANK_BACKWARD']) &
                                       (price_hist['RANK_NUM_TOUCH'] > 0) &
                                       (price_hist['ROLL_RANK_FORWARD'] == price_hist['RANK_NUM_TOUCH']),
                                    0.5, -1)

    
    #=================== ADJUST POINTS 2 ===================#
    '''
    Points based on the number of times prices REVERT or BREAK OUT of the range;
    At each time step, flag -1 if price is less than the range, 0 if it is within the range and 1 otherwise;
    FORWARD_TREND will be sum of the flag over the forward patience_time, and similarly for PREV_TREND;
    The product between FORWARDTREND and PREV_TREND >= 0 at the time price within the range --> Add 1 points to the range; otherwise -2.
    '''
    ranges = price_hist.copy().index
    
    for r in ranges:
        df_observe[r] = df_observe[price_type].apply(lambda x: -1 if x <= r.left
                                                else (0 if x in r
                                                        else 1
                                                        ))
        # # Create a forward rolling view
        # df_observe[f'{r}_FORWARD_TREND'] = df_observe[r].rolling(window = patience_time).sum()
    
        # # Create a backward rolling view by reversing the DataFrame and applying forward rolling
        # df_observe[f'{r}_PREV_TREND'] = df_observe[::-1][r].rolling(window = patience_time).sum()
    
        # Adjust points 2 based on revert and break out
        df_observe[f'{r}_REVERT_POINTS'] = np.where((df_observe[r] == 0) & (df_observe[r].rolling(window = patience_time).sum()*df_observe[::-1][r].rolling(window = patience_time).sum() >= 0),
                                            1,
                                            -2)
    
    adjust2 = pd.DataFrame(df_observe.filter(like = 'REVERT_POINTS').sum().rank(pct = True, method = 'first'))
    
    adjust2.index = ranges
    
    adjust2.columns = ['ADJUST_POINTS_2']
    
    price_hist = pd.concat([price_hist, adjust2], axis = 1)
    
    price_hist['SR_SCORE'] = price_hist['RANK_NUM_TOUCH'] + price_hist['ADJUST_POINTS_1'] + price_hist['ADJUST_POINTS_2']
    price_hist['SR_RANK'] = price_hist['SR_SCORE'].rank(ascending = False)
    
    price_hist = price_hist.drop(columns = ['ROLL_RANK_FORWARD', 'ROLL_RANK_BACKWARD'])

    return(price_hist)


def plot_sr(df_observe, sr_range, patience_range, patience_time, max_num_sr, cutoff, type, path, open_tab):
    price_hist = prepare_df_sr(df_observe = df_observe.copy(),
                               sr_range = sr_range,
                               patience_range = patience_range,
                               patience_time = patience_time,
                               type = type,
                               )

    fig = plot_df(df_observe.copy(), path = None, open_tab = False)

    for idx, i in enumerate(price_hist.index):
        if price_hist.loc[i, 'SR_RANK'] <= max_num_sr and price_hist.loc[i, 'SR_SCORE'] >= cutoff:
            uprange = dict(type='line',
                                x0=df_observe.index.min(),
                                x1=df_observe.index.max(),
                                y0=price_hist.index[idx].right,
                                y1=price_hist.index[idx].right,
                                line=dict(color='white', width=1, dash='dash'))

            lowrange = dict(type='line',
                            x0=df_observe.index.min(),
                            x1=df_observe.index.max(),
                            y0=price_hist.index[idx].left,
                            y1=price_hist.index[idx].left,
                            line=dict(color='white', width=1, dash='dash'))

            fig.add_shape(uprange,
                        row=1,
                        col=1)

            fig.add_shape(lowrange,
                        row=1,
                        col=1)

            fig.add_shape(
                type='rect',
                x0=df_observe.index.min(),
                x1=df_observe.index.max(),
                y0=price_hist.index[idx].left,
                y1=price_hist.index[idx].right,
                fillcolor='rgba(200, 160, 255, 0.2)',  # Light purple color with opacity
                line=dict(color='rgba(255, 255, 255, 0)'),  # Set border color and opacity
                row=1,
                col=1
            )



    # sum_score = price_hist.loc[(price_hist['SR_RANK'] <= max_num_sr) & (price_hist['SR_SCORE'] >= cutoff), 'SR_SCORE'].sum()

    # fig.add_annotation(
    # text=sum_score,
    # x=df_observe.index.min(),  # X-coordinate of the text box
    # y=df_observe['HIGH'].max(),  # Y-coordinate of the text box
    # showarrow=True,
    # arrowhead=7,
    # ax=0,
    # ay=-40
    # )

    if path:
        # Write HTML output
        fig.write_html(path)
        url = path
        if open_tab:
            webbrowser.open(url, new=2)  # open in new tab

    return(fig)


def sr_score(param, df_observe):
    price_hist = prepare_df_sr(sr_range = param['sr_range'],
                    patience_range = param['patience_range'],
                    patience_time = param['patience_time'],
                    df_observe = df_observe.copy(), 
                    type = param['type']
                    )

    sum_score = price_hist.loc[(price_hist['SR_RANK'] <= param['max_num_range']) & (price_hist['SR_SCORE'] >= param['cutoff']), 'SR_SCORE'].sum()
    return(sum_score)


def search_sr(df_observe, sr_arr, pr_arr, pt_arr, max_num_arr, cutoff_arr, type):

    space = {
        "sr_range": hp.choice("sr_range", sr_arr),
        "patience_range": hp.choice("patience_range", pr_arr),
        "patience_time": hp.choice("patience_time", pt_arr),
        "max_num_range": hp.choice("max_num_range", max_num_arr),
        "cutoff": hp.choice("cutoff", cutoff_arr),
        "type": type
    }


    def hyperparameter_tuning(params):
        score = sr_score(params, df_observe.copy())
        return {"loss": -score, "status": STATUS_OK}

    # Initialize trials object
    trials = Trials()

    best = fmin(
        fn=hyperparameter_tuning,
        space = space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )

    best['patience_range'] = pr_arr[best['patience_range']]
    best['patience_time'] = pt_arr[best['patience_time']]
    best['max_num_range'] = max_num_arr[best['max_num_range']]
    best['sr_range'] = sr_arr[best['sr_range']]
    best['cutoff'] = cutoff_arr[best['cutoff']]
    best['type'] = type
    
    return(best)

#=======================LOCAL EXTREMUM/ SWING HIGHS SWING LOWS=======================

def search_extremum(order, df_observe):
    
    local_max_indices = argrelmax(data = df_observe.iloc[order: -order, :]['HIGH'].values, axis = 0, order = order)[0]
    local_min_indices = argrelmin(data = df_observe.iloc[order: -order, :]['LOW'].values, axis = 0, order = order)[0]

    return(local_max_indices, local_min_indices)


def swing_highs_lows(df, swing_length = 50):
    """
    INTERCHANGABLE WITH search_extremum

    Swing Highs and Lows
    A swing high is when the current high is the highest high out of the swing_length amount of candles before and after.
    A swing low is when the current low is the lowest low out of the swing_length amount of candles before and after.

    parameters:
    swing_length: int - the amount of candles to look back and forward to determine the swing high or low

    returns:
    HighLow = 1 if swing high, -1 if swing low
    Level = the level of the swing high or low
    """

    swing_length *= 2
    # set the highs to 1 if the current high is the highest high in the last 5 candles and next 5 candles
    swing_highs_lows = np.where(
        df["HIGH"]
        == df["HIGH"].shift(-(swing_length // 2)).rolling(swing_length).max(),
        1,
        np.where(
            df["LOW"]
            == df["LOW"].shift(-(swing_length // 2)).rolling(swing_length).min(),
            -1,
            np.nan,
        ),
    )

    while True:
        positions = np.where(~np.isnan(swing_highs_lows))[0]

        if len(positions) < 2:
            break

        current = swing_highs_lows[positions[:-1]]
        next = swing_highs_lows[positions[1:]]

        highs = df["HIGH"].iloc[positions[:-1]].values
        lows = df["LOW"].iloc[positions[:-1]].values

        next_highs = df["HIGH"].iloc[positions[1:]].values
        next_lows = df["LOW"].iloc[positions[1:]].values

        index_to_remove = np.zeros(len(positions), dtype=bool)

        consecutive_highs = (current == 1) & (next == 1)
        index_to_remove[:-1] |= consecutive_highs & (highs < next_highs)
        index_to_remove[1:] |= consecutive_highs & (highs >= next_highs)

        consecutive_lows = (current == -1) & (next == -1)
        index_to_remove[:-1] |= consecutive_lows & (lows > next_lows)
        index_to_remove[1:] |= consecutive_lows & (lows <= next_lows)

        if not index_to_remove.any():
            break

        swing_highs_lows[positions[index_to_remove]] = np.nan

    positions = np.where(~np.isnan(swing_highs_lows))[0]

    if len(positions) > 0:
        if swing_highs_lows[positions[0]] == 1:
            swing_highs_lows[0] = -1
        if swing_highs_lows[positions[0]] == -1:
            swing_highs_lows[0] = 1
        if swing_highs_lows[positions[-1]] == -1:
            swing_highs_lows[-1] = 1
        if swing_highs_lows[positions[-1]] == 1:
            swing_highs_lows[-1] = -1

    level = np.where(
        ~np.isnan(swing_highs_lows),
        np.where(swing_highs_lows == 1, df["HIGH"], df["LOW"]),
        np.nan,
    )

    df_swing = pd.concat(
        [
            pd.Series(swing_highs_lows, name="HIGHLOW"),
            pd.Series(level, name="LEVEL"),
        ],
        axis=1,
    )
    df_swing.index = df.index
    # df_swing = df_swing.dropna()


    return(df_swing)


def plot_extremum(df_observe, order, path, open_tab):

    local_max_indices, local_min_indices = search_extremum(df_observe = df_observe.copy(), order = order)
    fig = plot_df(df_observe.copy(), path = None, open_tab = False)
    
    for id in local_max_indices:
        fig.add_annotation(
            text='Local max',
            x=df_observe.index[id + order],  # X-coordinate of the text box
            y=df_observe.iloc[id + order,:]['HIGH'],  # Y-coordinate of the text box
            showarrow=True,
            arrowhead=1,
            arrowcolor = 'red',
            arrowside = 'end',
            opacity = 1,
            ax=0,
            ay=-45,
            font=dict(
            family="Arial, sans-serif",
            size=10,
            color="red",
            # bold=True
            )
        )
    
    for id in local_min_indices:
        fig.add_annotation(
            text='Local min',
            x=df_observe.index[id + order],  # X-coordinate of the text box
            y=df_observe.iloc[id + order,:]['LOW'],  # Y-coordinate of the text box
            showarrow=True,
            arrowhead=1,
            arrowcolor = 'red',
            arrowside = 'end',
            opacity = 1,
            ax=0,
            ay=45,
            font=dict(
            family="Arial, sans-serif",
            size=10,
            color="red",
            # bold=True
            )
        )
    
    if path:
        # Write HTML output
        fig.write_html(path)
        url = path
        if open_tab:
            webbrowser.open(url, new=2)  # open in new tab

    return(fig)


def bos_choch(df, df_swing_highs_lows, close_break = True):
    """
    BREAK OF PREVIOUS SWING HIGHS, LOWS
    

    BOS - Break of Structure
    CHoCH - Change of Character
    these are both indications of market structure changing

    parameters:
    df_swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
    close_break: bool - if True then the break of structure will be mitigated based on the close of the candle otherwise it will be the high/low.

    returns:
    BOS = 1 if bullish break of structure, -1 if bearish break of structure
    CHOCH = 1 if bullish change of character, -1 if bearish change of character
    Level = the level of the break of structure or change of character
    BrokenIndex = the index of the candle that broke the level
    """

    df_swing_highs_lows = df_swing_highs_lows.copy()

    level_order = []
    highs_lows_order = []

    bos = np.zeros(len(df), dtype=np.int32)
    choch = np.zeros(len(df), dtype=np.int32)
    level = np.zeros(len(df), dtype=np.float32)

    last_positions = []

    for i in range(len(df_swing_highs_lows["HIGHLOW"])):
        if not np.isnan(df_swing_highs_lows["HIGHLOW"][i]):
            level_order.append(df_swing_highs_lows["LEVEL"][i])
            highs_lows_order.append(df_swing_highs_lows["HIGHLOW"][i])
            if len(level_order) >= 4:
                # bullish bos
                bos[last_positions[-2]] = (
                    1
                    if (
                        np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                        and np.all(
                            level_order[-4]
                            < level_order[-2]
                            < level_order[-3]
                            < level_order[-1]
                        )
                    )
                    else 0
                )
                level[last_positions[-2]] = (
                    level_order[-3] if bos[last_positions[-2]] != 0 else 0
                )

                # bearish bos
                bos[last_positions[-2]] = (
                    -1
                    if (
                        np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                        and np.all(
                            level_order[-4]
                            > level_order[-2]
                            > level_order[-3]
                            > level_order[-1]
                        )
                    )
                    else bos[last_positions[-2]]
                )
                level[last_positions[-2]] = (
                    level_order[-3] if bos[last_positions[-2]] != 0 else 0
                )

                # bullish choch
                choch[last_positions[-2]] = (
                    1
                    if (
                        np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                        and np.all(
                            level_order[-1]
                            > level_order[-3]
                            > level_order[-4]
                            > level_order[-2]
                        )
                    )
                    else 0
                )
                level[last_positions[-2]] = (
                    level_order[-3]
                    if choch[last_positions[-2]] != 0
                    else level[last_positions[-2]]
                )

                # bearish choch
                choch[last_positions[-2]] = (
                    -1
                    if (
                        np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                        and np.all(
                            level_order[-1]
                            < level_order[-3]
                            < level_order[-4]
                            < level_order[-2]
                        )
                    )
                    else choch[last_positions[-2]]
                )
                level[last_positions[-2]] = (
                    level_order[-3]
                    if choch[last_positions[-2]] != 0
                    else level[last_positions[-2]]
                )

            last_positions.append(i)

    broken = np.zeros(len(df), dtype=np.int32)
    for i in np.where(np.logical_or(bos != 0, choch != 0))[0]:
        mask = np.zeros(len(df), dtype=np.bool_)
        # if the bos is 1 then check if the candles high has gone above the level
        if bos[i] == 1 or choch[i] == 1:
            mask = df["CLOSE" if close_break else "HIGH"][i + 2 :] > level[i]
        # if the bos is -1 then check if the candles low has gone below the level
        elif bos[i] == -1 or choch[i] == -1:
            mask = df["CLOSE" if close_break else "LOW"][i + 2 :] < level[i]
        if np.any(mask):
            j = np.argmax(mask) + i + 2
            broken[i] = j
            # if there are any unbroken bos or choch that started before this one and ended after this one then remove them
            for k in np.where(np.logical_or(bos != 0, choch != 0))[0]:
                if k < i and broken[k] >= j:
                    bos[k] = 0
                    choch[k] = 0
                    level[k] = 0

    # remove the ones that aren't broken
    for i in np.where(
        np.logical_and(np.logical_or(bos != 0, choch != 0), broken == 0)
    )[0]:
        bos[i] = 0
        choch[i] = 0
        level[i] = 0

    # replace all the 0s with np.nan
    bos = np.where(bos != 0, bos, np.nan)
    choch = np.where(choch != 0, choch, np.nan)
    level = np.where(level != 0, level, np.nan)
    broken = np.where(broken != 0, broken, np.nan)

    bos = pd.Series(bos, name="BOS")
    choch = pd.Series(choch, name="CHOCH")
    level = pd.Series(level, name="LEVEL")
    broken = pd.Series(broken, name="BROKEN_TIME")

    df_bos_choch = pd.concat([bos, choch, level, broken], axis=1)
    df_bos_choch.index = df.index

    df_bos_choch['BROKEN_TIME'] = df_bos_choch['BROKEN_TIME'].apply(lambda x: df_bos_choch.index[int(x)] if pd.notnull(x) and x != 0 else np.nan)
    df_bos_choch['FLAG_BROKEN'] = df_bos_choch['BROKEN_TIME'].apply(lambda x: 1 if pd.notnull(x) else 0)
        

    return(df_bos_choch) 

#=======================Fair value gap=======================

def fvg(df, join_consecutive=False):
        """
        FVG - Fair Value Gap
        A fair value gap is when the previous high is lower than the next low if the current candle is bullish.
        Or when the previous low is higher than the next high if the current candle is bearish.

        parameters:
        join_consecutive: bool - if there are multiple FVG in a row then they will be merged into one using the highest top and the lowest bottom

        returns:
        FVG = 1 if bullish fair value gap, -1 if bearish fair value gap
        Top = the top of the fair value gap
        Bottom = the bottom of the fair value gap
        MitigatedIndex = the index of the candle that mitigated the fair value gap
        """

        fvg = np.where(
            (
                (df["HIGH"].shift(1) < df["LOW"].shift(-1))
                & (df["CLOSE"] > df["OPEN"])
            )
            | (
                (df["LOW"].shift(1) > df["HIGH"].shift(-1))
                & (df["CLOSE"] < df["OPEN"])
            ),
            np.where(df["CLOSE"] > df["OPEN"], 1, -1),
            np.nan,
        )

        top = np.where(
            ~np.isnan(fvg),
            np.where(
                df["CLOSE"] > df["OPEN"],
                df["LOW"].shift(-1),
                df["LOW"].shift(1),
            ),
            np.nan,
        )

        bottom = np.where(
            ~np.isnan(fvg),
            np.where(
                df["CLOSE"] > df["OPEN"],
                df["HIGH"].shift(1),
                df["HIGH"].shift(-1),
            ),
            np.nan,
        )

        # if there are multiple consecutive fvg then join them together using the highest top and lowest bottom and the last index
        if join_consecutive:
            for i in range(len(fvg) - 1):
                if fvg[i] == fvg[i + 1]:
                    top[i + 1] = max(top[i], top[i + 1])
                    bottom[i + 1] = min(bottom[i], bottom[i + 1])
                    fvg[i] = top[i] = bottom[i] = np.nan

        mitigated_index = np.zeros(len(df), dtype=np.int32)
        for i in np.where(~np.isnan(fvg))[0]:
            mask = np.zeros(len(df), dtype=np.bool_)
            if fvg[i] == 1:
                mask = df["LOW"][i + 2 :] <= top[i]
            elif fvg[i] == -1:
                mask = df["HIGH"][i + 2 :] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated_index[i] = j

        mitigated_index = np.where(np.isnan(fvg), np.nan, mitigated_index)

        df_fvg = pd.concat(
                [
                    pd.Series(fvg, name="FVG"),
                    pd.Series(top, name="TOP_FVG"),
                    pd.Series(bottom, name="BOTTOM_FVG"),
                    pd.Series(mitigated_index, name="MITIGATED_TIME_FVG"),
                ],
                axis = 1,
                    )
        df_fvg.index = df.index
        df_fvg['MITIGATED_TIME_FVG'] = df_fvg['MITIGATED_TIME_FVG'].apply(lambda x: df_fvg.index[int(x)] if pd.notnull(x) and x != 0 else np.nan)
        df_fvg['FLAG_MITIGATED_FVG'] = df_fvg['MITIGATED_TIME_FVG'].apply(lambda x: 1 if pd.notnull(x) else 0)
        
        return(df_fvg)


#=======================Liquidity=======================

def liquidity(df, df_swing_highs_lows, range_percent = 0.01):
    """
    Liquidity
    Liquidity is when there are multiple highs within a small range of each other,
    or multiple lows within a small range of each other.

    parameters:
    swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
    range_percent: float - the percentage of the range to determine liquidity

    returns:
    Liquidity = 1 if bullish liquidity, -1 if bearish liquidity
    Level = the level of the liquidity
    End = the index of the last liquidity level
    Swept = the index of the candle that swept the liquidity
    """

    # Work on a copy so the original is not modified.
    shl = df_swing_highs_lows.copy()
    n = len(df)
    
    # Calculate the pip range based on the overall high-low range.
    pip_range = (df["HIGH"].max() - df["LOW"].min()) * range_percent

    # Preconvert required columns to numpy arrays.
    df_high = df["HIGH"].values
    df_low = df["LOW"].values
    # Make a copy to allow in-place marking of used candidates.
    shl_HL = shl["HIGHLOW"].values.copy()
    shl_Level = shl["LEVEL"].values.copy()

    # Initialise output arrays with NaN (to match later replacement of zeros).
    liquidity = np.full(n, np.nan, dtype=np.float32)
    liquidity_level = np.full(n, np.nan, dtype=np.float32)
    liquidity_end = np.full(n, np.nan, dtype=np.float32)
    liquidity_swept = np.full(n, np.nan, dtype=np.float32)

    # Process bullish liquidity (HighLow == 1)
    bull_indices = np.nonzero(shl_HL == 1)[0]
    for i in bull_indices:
        # Skip if this candidate has already been used.
        if shl_HL[i] != 1:
            continue
        high_level = shl_Level[i]
        range_low = high_level - pip_range
        range_high = high_level + pip_range
        group_levels = [high_level]
        group_end = i

        # Determine the swept index:
        # Find the first candle after i where the high reaches or exceeds range_high.
        c_start = i + 1
        if c_start < n:
            cond = df_high[c_start:] >= range_high
            if np.any(cond):
                swept = c_start + int(np.argmax(cond))
            else:
                swept = 0
        else:
            swept = 0

        # Iterate only over candidate indices greater than i.
        for j in bull_indices:
            if j <= i:
                continue
            # Emulate the inner loop break: if we've reached or passed the swept index, stop.
            if swept and j >= swept:
                break
            # If candidate j is within the liquidity range, add it and mark it as used.
            if shl_HL[j] == 1 and (range_low <= shl_Level[j] <= range_high):
                group_levels.append(shl_Level[j])
                group_end = j
                shl_HL[j] = 0  # mark candidate as used
        # Only record liquidity if more than one candidate is grouped.
        if len(group_levels) > 1:
            avg_level = sum(group_levels) / len(group_levels)
            liquidity[i] = 1
            liquidity_level[i] = avg_level
            liquidity_end[i] = group_end
            liquidity_swept[i] = swept

    # Process bearish liquidity (HighLow == -1)
    bear_indices = np.nonzero(shl_HL == -1)[0]
    for i in bear_indices:
        if shl_HL[i] != -1:
            continue
        low_level = shl_Level[i]
        range_low = low_level - pip_range
        range_high = low_level + pip_range
        group_levels = [low_level]
        group_end = i

        # Find the first candle after i where the low reaches or goes below range_low.
        c_start = i + 1
        if c_start < n:
            cond = df_low[c_start:] <= range_low
            if np.any(cond):
                swept = c_start + int(np.argmax(cond))
            else:
                swept = 0
        else:
            swept = 0

        for j in bear_indices:
            if j <= i:
                continue
            if swept and j >= swept:
                break
            if shl_HL[j] == -1 and (range_low <= shl_Level[j] <= range_high):
                group_levels.append(shl_Level[j])
                group_end = j
                shl_HL[j] = 0
        if len(group_levels) > 1:
            avg_level = sum(group_levels) / len(group_levels)
            liquidity[i] = -1
            liquidity_level[i] = avg_level
            liquidity_end[i] = group_end
            liquidity_swept[i] = swept

    # Convert arrays to Series with the proper names.
    liq_series = pd.Series(liquidity, name="LIQUIDITY")
    level_series = pd.Series(liquidity_level, name="LEVEL")
    end_series = pd.Series(liquidity_end, name="END_TIME")
    swept_series = pd.Series(liquidity_swept, name="SWEPT_TIME")

    df_liquid = pd.concat([liq_series, level_series, end_series, swept_series], axis=1)
    df_liquid.index = df.index

    df_liquid['END_TIME'] = df_liquid['END_TIME'].apply(lambda x: df_liquid.index[int(x)] if pd.notnull(x) and x != 0 else np.nan)
    df_liquid['SWEPT_TIME'] = df_liquid['SWEPT_TIME'].apply(lambda x: df_liquid.index[int(x)] if pd.notnull(x) and x != 0 else np.nan)

    return(df_liquid)
