import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)

import ta

import matplotlib.pyplot as plt

from src.support import *
from src.models_support import *

import tensorflow as tf
import tensorflow_hub as hub
from abc import ABC, abstractmethod
import keras
from keras import layers

plt.style.use('classic')

#======================= ALPHAs =======================#

class Base_Alpha:
    def __init__(self, strat, prepare_data):
        # Prepare function
        self.prepare_data = prepare_data
        # Strategy function
        self.strat = strat 

        self.config = None
        self.extra_params = {}

    def signal(self, df_observe):
        '''
        Return the tested dataframe with additional columns: signal (buy/ sell), stop loss, take profit.
        '''
        df_prepare = self.prepare_data(df_observe)           
        # Apply the strategy to the dataframe and return the result in IS
        df_result = df_prepare.copy().apply(lambda x: self.strat(x, **self.extra_params), axis=1)

        return(df_result)


class RSI_strat(Base_Alpha):
    def __init__(self, config):
        super().__init__(
            strat = self._RSI_strat,
            prepare_data = self._prepare_data
        )  # Pass the strategy function to the parent constructor
        
        '''
        config = {'RSI_PARAMS': {'INPUT_PARAM': 14, 'CUTOFF_BUY': 30, 'CUTOFF_SELL': 70}, 
                'base_SL': 10, 
                'base_TP': 20
                }
        '''
        self.RSI_param = config['RSI_PARAMS']
        self.base_SL = config['base_SL']
        self.base_TP = config['base_TP']
        
        
    def _prepare_data(self, df_observe):
        df_observe['RSI'] = ta.momentum.RSIIndicator(df_observe['CLOSE'], window = self.RSI_param['INPUT_PARAM']).rsi()
        return(df_observe)

    def _RSI_strat(self, row):
        '''Strategy for RSI, calculates the signal based on the RSI value.'''
        if row['RSI'] < self.RSI_param['CUTOFF_BUY']:
            return pd.Series([1, self.base_SL, self.base_TP], index=['SIGNAL', 'SL', 'TP']) # Buy signal
        elif row['RSI'] > self.RSI_param['CUTOFF_BUY']:
            return pd.Series([-1, self.base_SL, self.base_TP], index=['SIGNAL', 'SL', 'TP']) # Sell signal
        else:
            return pd.Series([0, 0, 0], index=['SIGNAL', 'SL', 'TP']) # No action (neutral)
        

class MCMC_strat_v1(Base_Alpha):
    def __init__(self, config):
        super().__init__(
            strat = self._MCMC_strat_v1,
            prepare_data = self._prepare_data
        )

        '''
        config = {
            'base_SL': 10, 
            'base_TP': 20,
            'DECISION_CUTOFF': {'ENTRY_BUY_CUTOFF': 1, 'ENTRY_SELL_CUTOFF': -1}
        }

        Conduct on simple transition 1-step matrix between the previous and current log return
        '''
        self.base_SL = config['base_SL']
        self.base_TP = config['base_TP']
        self.DECISION_CUTOFF = config['DECISION_CUTOFF']
        
        
    def _prepare_data(self, df_observe):
        #returns
        df_observe['RET(T)'] = 100*(df_observe['CLOSE'] - df_observe['CLOSE'].shift(1))/df_observe['CLOSE'].shift(1)
        return(df_observe)

    def _MCMC_strat_v1(self, row):
        if ((row['RET(T)'] > self.DECISION_CUTOFF['ENTRY_BUY_CUTOFF']) & (pd.isnull(row['RET(T)']) == False)):
            return pd.Series([1, self.base_SL, self.base_TP], index=['SIGNAL', 'SL', 'TP']) # Buy signal
        elif ((row['RET(T)'] < self.DECISION_CUTOFF['ENTRY_SELL_CUTOFF']) & (pd.isnull(row['RET(T)']) == False)):
            return pd.Series([-1, self.base_SL, self.base_TP], index=['SIGNAL', 'SL', 'TP']) # Sell signal
        else:
            return pd.Series([0, 0, 0], index=['SIGNAL', 'SL', 'TP']) # No action (neutral)
        

class TA_strat_v1(Base_Alpha):
    def __init__(self, config, extra_params):
        super().__init__(
            strat = self._TA_strat_v1,
            prepare_data = self._prepare_data
        )  # Pass the strategy function to the parent constructor
        
        '''
        config = {
            'RSI_PARAMS': {'INPUT_PARAM': 14, 'CUTOFF_BUY': 30, 'CUTOFF_SELL': 70},
            'BB_PARAMS': {'INPUT_MEAN': 50, 'INPUT_SD': 2, 'BB_CUTOFF': 5},
            'EMA_1_PARAMS': {'INPUT_PARAM': 50},
            'EMA_2_PARAMS': {'INPUT_PARAM': 200},
            'WEIGHT_DAILY': 5,
            'WEIGHT_4HOUR': 3,
            'WEIGHT_15MIN': 3,
            'LOOKBACK': 40,
            'base_SL': 10, 
            'base_TP': 20,
            'DECISION_CUTOFF': {'ENTRY_BUY_CUTOFF': 0.5, 'ENTRY_SELL_CUTOFF': 0.5}
        }

        extra_params = {
            'df_1_day_is': df_1_day_is, 
            'df_4_hour_is': df_4_hour_is, 
            'df_15_min_is': df_15_min_is
        }

        '''
        self.RSI_PARAMS = config['RSI_PARAMS']
        self.BB_PARAMS = config['BB_PARAMS']
        self.EMA_1_PARAMS = config['EMA_1_PARAMS']
        self.EMA_2_PARAMS = config['EMA_2_PARAMS']
        self.WEIGHT_DAILY = config['WEIGHT_DAILY']
        self.WEIGHT_4HOUR = config['WEIGHT_4HOUR']
        self.WEIGHT_15MIN = config['WEIGHT_15MIN']
        self.LOOKBACK = config['LOOKBACK']
        self.base_SL = config['base_SL']
        self.base_TP = config['base_TP']
        self.DECISION_CUTOFF = config['DECISION_CUTOFF']
        self.extra_params = extra_params
        
    def _prepare_data(self, df_observe):

        # RSI
        df_observe['RSI'] = ta.momentum.RSIIndicator(df_observe['CLOSE'], window = self.RSI_PARAMS['INPUT_PARAM']).rsi()
        
        # Bollinger band
        df_observe[f'BB_UPPER_BAND({self.BB_PARAMS["INPUT_MEAN"]})'] = ta.volatility.BollingerBands(df_observe['CLOSE'], 
                                                                                                    window = self.BB_PARAMS['INPUT_MEAN'], 
                                                                                                    window_dev = self.BB_PARAMS['INPUT_SD']).bollinger_hband()
        df_observe[f'BB_LOWER_BAND({self.BB_PARAMS["INPUT_MEAN"]})'] = ta.volatility.BollingerBands(df_observe['CLOSE'], 
                                                                                                    window = self.BB_PARAMS['INPUT_MEAN'], 
                                                                                                    window_dev = self.BB_PARAMS['INPUT_SD']).bollinger_lband()
        
        # Exponential moving average
        df_observe[f'EMA({self.EMA_1_PARAMS["INPUT_PARAM"]})'] = ta.trend.EMAIndicator(df_observe['CLOSE'],
                                                window = self.EMA_1_PARAMS["INPUT_PARAM"]).ema_indicator()
        df_observe[f'EMA({self.EMA_2_PARAMS["INPUT_PARAM"]})'] = ta.trend.EMAIndicator(df_observe['CLOSE'],
                                                window = self.EMA_2_PARAMS["INPUT_PARAM"]).ema_indicator()

        return(df_observe)
    
    def _identify_general_trend(self, df_observe):
        '''
        Identify trend and reversal based ONLY on support, resistance, previous highs and lows
        '''

        # Prepare local max, min

        daily_local_max_indices, daily_local_min_indices = search_extremum(df_observe = df_observe, order = 3)

        # Prepare support, resistance

        sr_arr = [5]
        pr_arr = [4]
        pt_arr = [3]
        max_num_arr = [5]
        cutoff_arr = [1]

        # best_params = search_sr(df_observe, 
        #                         sr_arr = sr_arr,
        #                         pr_arr = pr_arr,
        #                         pt_arr = pt_arr,
        #                         max_num_arr = max_num_arr,
        #                         cutoff_arr = cutoff_arr
        #                 )

        best_params = {
             'sr_range': 5,
             'patience_range': 4,
             'patience_time': 3,
             'max_num_range': 5,
             'cutoff': 1,
        }


        # price_hist = prepare_df_sr(df_observe = df_observe,
        #         sr_range = best_params['sr_range'],
        #         patience_range = best_params['patience_range'],
        #         patience_time = best_params['patience_time'])


        # sup_resist = price_hist[(price_hist['SR_RANK'] <= best_params['max_num_range']) & (price_hist['SR_SCORE'] >= best_params['cutoff'])].index
        dist_to_sr = []
        # for _, sr in enumerate(sup_resist):
        #         dist_to_sr.append(df_observe['CLOSE'][-1] - (sr.left + sr.right)/2)

        period_change = df_observe['CLOSE'][-1] - df_observe['CLOSE'][0]

        num_local_max = len(daily_local_max_indices)
        num_local_min = len(daily_local_min_indices)

        if num_local_max > 0:
                latest_local_max_change = df_observe['CLOSE'][-1] - df_observe.iloc[daily_local_max_indices[-1] + 2, :]['HIGH']
                local_max_change = df_observe.iloc[daily_local_max_indices[-1] + 2, :]['HIGH'] - df_observe.iloc[daily_local_max_indices[0] + 2, :]['HIGH']
        else:
                latest_local_max_change = 0
                local_max_change = 0

        if num_local_min > 0:
                latest_local_min_change = df_observe['CLOSE'][-1] - df_observe.iloc[daily_local_min_indices[-1] + 2, :]['LOW']
                local_min_change = df_observe.iloc[daily_local_min_indices[-1] + 2, :]['LOW'] - df_observe.iloc[daily_local_min_indices[0] + 2, :]['LOW']
        else:
                latest_local_min_change = 0
                local_min_change = 0

        trend_score = (np.abs(period_change) > 10)*np.sign(period_change) + \
                        (np.abs(period_change) > 30)*2*np.sign(period_change) + \
                        (num_local_max <= 3)*np.sign(period_change) + \
                        (num_local_min <= 3)*np.sign(period_change) + \
                        (num_local_max < 2)*2*np.sign(period_change) + \
                        (num_local_min < 2)*2*np.sign(period_change) + \
                        (np.abs(local_max_change) > 5)*np.sign(local_max_change) + \
                        (np.abs(local_max_change) > 10)*2*np.sign(local_max_change) + \
                        (np.abs(local_min_change) > 5)*np.sign(local_min_change) + \
                        (np.abs(local_min_change) > 10)*2*np.sign(local_min_change)
        trend_score /= 15

        pivot_score = (period_change > 0)*(latest_local_max_change < -10) + \
                        (period_change > 0)*(latest_local_max_change < -15)*2 + \
                        (period_change > 0)*(latest_local_min_change < 0) + \
                        (period_change > 0)*(latest_local_min_change < -10)*2 + \
                        (period_change > 0)*(latest_local_min_change < -15)*3 + \
                        (period_change < 0)*(latest_local_min_change > 10) + \
                        (period_change < 0)*(latest_local_min_change > 15)*2 + \
                        (period_change < 0)*(latest_local_max_change < 0) + \
                        (period_change < 0)*(latest_local_max_change < -10)*2 + \
                        (period_change < 0)*(latest_local_max_change < -15)*3
        
        if len(dist_to_sr) >= 2:
            pivot_score += (period_change > 0)*(dist_to_sr[-1] < -10) + \
                        (period_change > 0)*(dist_to_sr[-1] < -15)*2 + \
                        (period_change < 0)*(dist_to_sr[0] > 10) + \
                        (period_change < 0)*(dist_to_sr[0] > 15)*2
            pivot_score /= 12
        else: 
            pivot_score /= 9

        return(trend_score, pivot_score)

    def _TA_strat_v1(self, row, df_1_day_is, df_4_hour_is, df_15_min_is):
        df_1_day_is_observed = df_1_day_is[df_1_day_is.index <= row.name].tail(self.LOOKBACK).copy()
        df_4_hour_is_observed = df_4_hour_is[df_4_hour_is.index <= row.name].tail(self.LOOKBACK).copy()
        df_15_min_is_observed = df_15_min_is[df_15_min_is.index <= row.name].tail(self.LOOKBACK).copy()

        trend_score_daily, pivot_score_daily = self._identify_general_trend(df_1_day_is_observed.copy())
        trend_score_4hour, pivot_score_4hour = self._identify_general_trend(df_4_hour_is_observed.copy())
        trend_score_15min, pivot_score_15min = self._identify_general_trend(df_15_min_is_observed.copy())

        entry_buy_score = (
                (trend_score_daily > 0.5) * self.WEIGHT_DAILY + \
                ((trend_score_daily < -0.5) & (pivot_score_daily > 0.5)) * self.WEIGHT_DAILY + \
                (trend_score_4hour > 0.3) * self.WEIGHT_4HOUR + \
                ((trend_score_4hour < -0.3) & (pivot_score_4hour > 0.3)) * self.WEIGHT_4HOUR + \
                (trend_score_15min > 0) * self.WEIGHT_15MIN + \
                1
            ) * \
            (
                (row['RSI'] < self.RSI_PARAMS['CUTOFF_BUY']) + \
                (
                    (row['CLOSE'] <= row[f'EMA({self.EMA_1_PARAMS["INPUT_PARAM"]})']) & \
                    (row['CLOSE'] >= row[f'EMA({self.EMA_2_PARAMS["INPUT_PARAM"]})'])
                ) + \
                (np.abs(row['CLOSE'] - row[f'BB_LOWER_BAND({self.BB_PARAMS["INPUT_MEAN"]})']) < self.BB_PARAMS['BB_CUTOFF']) 
            )

        entry_buy_score /= (2*self.WEIGHT_DAILY + 2*self.WEIGHT_4HOUR + self.WEIGHT_15MIN + 1)*(3)

        entry_sell_score = (
                (trend_score_daily < -0.5) * self.WEIGHT_DAILY + \
                ((trend_score_daily > 0.5) & (pivot_score_daily > 0.5)) * self.WEIGHT_DAILY + \
                (trend_score_4hour < -0.3) * self.WEIGHT_4HOUR + \
                ((trend_score_4hour > 0.3) & (pivot_score_4hour > 0.3)) * self.WEIGHT_4HOUR + \
                (trend_score_15min < 0) * self.WEIGHT_15MIN + \
                1
            ) * \
            (
                (row['RSI'] > self.RSI_PARAMS['CUTOFF_SELL']) + \
                (
                    (row['CLOSE'] >= row[f'EMA({self.EMA_1_PARAMS["INPUT_PARAM"]})']) & \
                    (row['CLOSE'] <= row[f'EMA({self.EMA_2_PARAMS["INPUT_PARAM"]})'])
                ) + \
                (np.abs(row['CLOSE'] - row[f'BB_UPPER_BAND({self.BB_PARAMS["INPUT_MEAN"]})']) < self.BB_PARAMS['BB_CUTOFF']) 
            )

        entry_sell_score /= (2*self.WEIGHT_DAILY + 2*self.WEIGHT_4HOUR + self.WEIGHT_15MIN + 1)*(3)
        
        signal = 0
        if ((entry_buy_score > self.DECISION_CUTOFF['ENTRY_BUY_CUTOFF']) & (entry_buy_score > entry_sell_score)):
            signal = 1
            SL = self.base_SL
            TP = self.base_TP
        elif ((entry_sell_score > self.DECISION_CUTOFF['ENTRY_SELL_CUTOFF']) & (entry_sell_score > entry_buy_score)):
            signal = -1
            SL = self.base_SL
            TP = self.base_TP
        else: 
            signal = 0
            SL = 0
            TP = 0
        

        return(pd.Series([trend_score_daily, 
                        pivot_score_daily, 
                        trend_score_4hour, 
                        pivot_score_4hour, 
                        trend_score_15min, 
                        pivot_score_15min, 
                        entry_buy_score, 
                        entry_sell_score,
                        signal,
                        SL,
                        TP
                        ], 
                        index = [
                            'TREND_SCORE_DAILY',
                            'PIVOT_SCORE_DAILY',
                            'TREND_SCORE_4HOUR',
                            'PIVOT_SCORE_4HOUR',
                            'TREND_SCORE_15MIN',
                            'PIVOT_SCORE_15MIN',
                            'ENTRY_BUY_SCORE',
                            'ENTRY_SELL_SCORE',
                            'SIGNAL', 
                            'SL',
                            'TP'
                        ]))
       


#======================= GAMMAs =======================#

# Bert model for news extracting



class GAMMA_BERT(keras.Model, ABC):
    def __init__(self,
            tfhub_handle_encoder,
            tfhub_handle_preprocess,
            loss_weight,
            **kwargs):
        super(GAMMA_BERT, self).__init__(**kwargs)

        self.tfhub_handle_preprocess = tfhub_handle_preprocess
        self.tfhub_handle_preprocess = tfhub_handle_preprocess
        self.loss_weight = loss_weight
        
        self.total_loss_tracker = keras.metrics.Mean(name = "total_loss")

        # binarycrossentropy distance between true and predicted FLAG_HIGH_RISK
        self.binarycrossentropy_tracker = keras.metrics.Mean(name = "binarycrossentropy")

        # mse distance between true and predicted ratio_mean_A_B
        self.mse_mean_tracker = keras.metrics.Mean(name = "mse_mean")
        
        # mse distance between true and predicted ratio_var_A_B
        self.mse_var_tracker = keras.metrics.Mean(name = "mse_var")
        
        self.bert = self._get_bert()
        

    def call(self, X):
        results = self.bert(tf.constant([X]))
        return results


    def _get_bert(self):
        text_input = tf.keras.layers.Input(shape = (), dtype = tf.string, name = 'text_input')
        preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name = 'text_preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable = True, name = 'BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.3)(net)
        FLAG_HIGH_RISK = tf.keras.layers.Dense(1, activation = None, name = 'FLAG_high_risk')(net)
        RATIO_MEAN = tf.keras.layers.Dense(1, activation = None, name = 'RATIO_MEAN')(net)
        RATIO_VAR = tf.keras.layers.Dense(1, activation = 'relu', name = 'RATIO_VAR')(net)
        
        bert = tf.keras.Model(text_input, [FLAG_HIGH_RISK, RATIO_MEAN, RATIO_VAR])

        return(bert)


    def summary(self):
        self.bert.summary()
        

    def train_step(self, X, y):
        
        flag_high_risk, ratio_mean, ratio_var = y
        
        with tf.GradientTape() as tape:

            predicted_flag_high_risk, predicted_ratio_mean, predicted_ratio_var = self.bert(X)

            binarycrossentropy = keras.losses.BinaryCrossentropy(from_logits = True)(flag_high_risk, predicted_flag_high_risk)
            mse_mean = tf.keras.losses.MeanSquareError()(ratio_mean, predicted_ratio_mean)
            mse_var = tf.keras.losses.MeanSquareError()(ratio_var, predicted_ratio_var)
            
            total_loss = self.loss_weight[0] * binarycrossentropy \
                            + self.loss_weight[1] * mse_mean \
                            + self.loss_weight[2] * mse_var
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.binarycrossentropy_tracker.update_state(binarycrossentropy)
        self.mse_mean_tracker.update_state(mse_mean)
        self.mse_var_tracker.update_state(mse_var)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss_1": self.binarycrossentropy_tracker.result(),
            "reconstruction_loss_2": self.mse_mean_tracker.result(),
            "reconstruction_loss_3": self.mse_var_tracker.result()
        }