import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 999)

import ta
import matplotlib.pyplot as plt

plt.style.use('classic')
from datetime import datetime, timedelta


class Base_Asset:
    def __init__(self, asset = 'XAUUSD'):
        self.asset = asset
        
    def _cal_margins(self):
        '''
        margins_level = 1/100 #1/200 #1/1000 #1/2000
        '''
        if self.asset == 'XAUUSD':
            # Base stats
            self.lot_per_asset = 0.01



class Backtest_report:
    def __init__(self, alpha, 
                 df_is,
                 base_SL = 10, base_TP = 20, 
                 max_existing_positions = 3, init_vol = 0.01, incre_vol = 0.01, max_vol = 0.1,
                 init_cap = 1000, incre_cap = 2, asset = 'XAUUSD', margins_level = 1/100, 
                 re_allocation = True
                 ):
        
        self.alpha = alpha # alpha class
        self.df_is = df_is # data
        self.init_vol = init_vol # initial volume
        self.re_allocation = re_allocation
        self.incre_vol = incre_vol # increment of volume
        self.max_vol = max_vol # max value of volume
        self.init_cap = init_cap # init capital
        self.incre_cap = incre_cap # the condition of capital if increasing the trading volume
        
        self.asset = Base_Asset(asset = asset)
        self.asset._cal_margins()
        self.margins_level = margins_level

        self.max_existing_positions = max_existing_positions
        self.base_SL = base_SL
        self.base_TP = base_TP    

        self.df_position_tracking = None
        self.df_balance_tracking = None
        self.df_balance_tracking_new = None
         

    def _track_positions(self, df_is, alpha):

        df_signal = alpha.signal(df_is)
        df_position_tracking = df_is[['CLOSE']].copy().merge(df_signal, how = 'left', on = 'DATE_TIME')

        df_position_tracking['VOL'] = self.init_vol
        signals = df_position_tracking[df_position_tracking['SIGNAL'] != 0]

        # Determine the time hitting TP/ SL
        for ids in range(len(signals)):
            s = signals.iloc[ids, :]
            df_temp = df_position_tracking.loc[(df_position_tracking.index > s.name) 
                    & (((df_position_tracking['CLOSE'] - s.CLOSE)*s.SIGNAL < s.SL*(-1)) | ((df_position_tracking['CLOSE'] - s.CLOSE)*s.SIGNAL > s.TP)), 
                    :]
            if len(df_temp) != 0:
                df_position_tracking.loc[s.name, 'TIME_CLOSE_POSITION'] = df_temp.index[0]
            else:
                df_position_tracking.loc[s.name, 'TIME_CLOSE_POSITION'] = pd.NaT

        signals = df_position_tracking[df_position_tracking['SIGNAL'] != 0]

        # Create the column to determine valid position (total number of positions, margins)
        df_position_tracking['FLAG_VALID_POSITION'] = np.where(df_position_tracking['SIGNAL'] != 0, 1, 0)
        df_position_tracking['USED_MARGINS'] = -1 * df_position_tracking['VOL'] * np.abs(df_position_tracking['SIGNAL']) * self.margins_level * df_position_tracking['CLOSE'] / self.asset.lot_per_asset

        # Adjust the FLAG_VALID_POSITION according to max_existing_positions
        existing_positions = signals[:self.max_existing_positions]

        for ids in range(len(signals)):
            if ids >= self.max_existing_positions:
                s = signals.iloc[ids, :]

                if (s.name < existing_positions['TIME_CLOSE_POSITION'].min()) & (len(existing_positions) == self.max_existing_positions):
                    df_position_tracking.loc[s.name, 'FLAG_VALID_POSITION'] = 0
                else:
                    df_position_tracking.loc[s.name, 'FLAG_VALID_POSITION'] = 1
        
                    existing_positions = existing_positions.loc[existing_positions['TIME_CLOSE_POSITION'] > s.name, :]
                    existing_positions = pd.concat([existing_positions, pd.DataFrame(s).transpose()], axis = 0)


        df_position_tracking['TIME_CLOSE_POSITION'] = np.where(df_position_tracking['FLAG_VALID_POSITION'] == 1, df_position_tracking['TIME_CLOSE_POSITION'], pd.NaT)
        df_position_tracking['TIME_CLOSE_POSITION'] = pd.to_datetime(df_position_tracking['TIME_CLOSE_POSITION'])

        df_position_tracking = df_position_tracking.merge(df_position_tracking[['CLOSE']], 
                                                how = 'left', 
                                                left_on = 'TIME_CLOSE_POSITION', 
                                                right_index = True,
                                                suffixes = ('_open', '_close')
                                                )

        # Calculate PNL
        df_position_tracking['PNL'] = (df_position_tracking['CLOSE_close'] - df_position_tracking['CLOSE_open'])*df_position_tracking['SIGNAL']*df_position_tracking['FLAG_VALID_POSITION']*(self.init_vol*100)
        df_position_tracking['PNL'] = np.where(df_position_tracking['PNL'] < self.base_SL*-1*(self.init_vol*100), self.base_SL*-1*(self.init_vol*100), 
                                    np.where(df_position_tracking['PNL'] > self.base_TP*(self.init_vol*100), self.base_TP*(self.init_vol*100), 
                                            np.where(df_position_tracking['PNL'].isnull() == True, 0, df_position_tracking['PNL'])))
        df_position_tracking['PNL'] = np.where(df_position_tracking['FLAG_VALID_POSITION'] == 1, df_position_tracking['PNL'], 0)
        

        # Re-indexing the data

        df_position_tracking = df_position_tracking.drop(columns=[col for col in df_position_tracking.columns if '_SCORE' in col])

        df_position_tracking = df_position_tracking.reset_index()
        df_position_tracking.columns = ['TIME_OPEN_POSITION', 'CLOSE_open', 
                            'SIGNAL', 'SL', 'TP', 'VOL', 'TIME_CLOSE_POSITION', 
                            'FLAG_VALID_POSITION', 'USED_MARGINS', 'CLOSE_close', 'PNL']

        df_position_tracking['POSITION_ID'] = df_position_tracking['TIME_OPEN_POSITION'].view('int64')//10**9

        df_position_tracking = df_position_tracking.loc[(df_position_tracking['FLAG_VALID_POSITION'] == 1) 
                                & (df_position_tracking['TIME_CLOSE_POSITION'].isnull() == False), :]
        
        return(df_position_tracking)
        
    def _cal_balance(self, df_is, df_position_tracking, 
                     cols_names = ['TIME_OPEN_POSITION', 'VOL', 'USED_MARGINS', 'POSITION_ID', 'TIME_CLOSE_POSITION', 'PNL']
                     ):
        # Calculate balance

        df_tracking = pd.DataFrame(index = df_is.index)

        df_tracking = df_tracking.merge(
            pd.pivot_table(
                df_position_tracking,
                index = 'TIME_OPEN_POSITION',
                values = [cols_names[1], cols_names[2], cols_names[3]],
                aggfunc = 'sum'
            ),
            how = 'left',
            left_index = True,
            right_index = True
        )

        df_tracking.columns = ['OPEN_POSITION_ID', 'USED_MARGINS', 'OPEN_VOL']

        df_tracking = df_tracking.merge(
            pd.pivot_table(
                df_position_tracking,
                index = cols_names[4],
                values = [cols_names[1], cols_names[2], cols_names[5]],
                aggfunc = 'sum'
            ),
            how = 'left',
            left_index = True,
            right_index = True
        )

        df_tracking.columns = ['OPEN_POSITION_ID', 'USED_MARGINS', 'OPEN_VOL', 'PNL', 'ADDITIONAL_MARGINS', 'CLOSED_VOL']
        df_tracking = df_tracking.fillna(0)

        df_tracking['OPEN_POSITION_ID'] = df_tracking['OPEN_POSITION_ID'].astype(str)
        df_tracking['CLOSED_VOL'] = df_tracking['CLOSED_VOL']*-1
        df_tracking['ADDITIONAL_MARGINS'] = df_tracking['ADDITIONAL_MARGINS']*-1

        df_tracking['NUM_CURRENT_POSITIONS'] = df_tracking['OPEN_VOL'].cumsum() + df_tracking['CLOSED_VOL'].cumsum()
        df_tracking['NUM_CURRENT_POSITIONS'] = df_tracking['NUM_CURRENT_POSITIONS'].astype(float)

        df_tracking['BALANCE'] = df_tracking['PNL'].cumsum() + self.init_cap
        df_tracking['FREE_MARGINS'] = df_tracking['BALANCE'] + df_tracking['USED_MARGINS'].cumsum() + df_tracking['ADDITIONAL_MARGINS'].cumsum()
        df_tracking['DRAWDOWN'] = np.where(df_tracking['PNL'] < 0, df_tracking['PNL']/(df_tracking['BALANCE'] - df_tracking['PNL']), 0)
        

        return(df_tracking)

    def _re_balance(self, df_position_tracking, df_balance_tracking):
        # Reallocating the capital
        
        idt = self.df_balance_tracking.index[0]
        cap = self.init_cap
        vol = self.init_vol

        df_position_tracking['NEW_VOL'] = df_position_tracking['VOL']
        df_position_tracking['NEW_USED_MARGINS'] = df_position_tracking['USED_MARGINS']
        df_position_tracking['NEW_PNL'] = df_position_tracking['PNL']

        for idd, d in enumerate(df_balance_tracking.index): 
            if df_balance_tracking.loc[d, 'BALANCE']/cap >= self.incre_cap:
                idt =  d
                cap *= self.incre_cap
                vol += self.incre_vol

                if vol >= self.max_vol:
                    vol = self.max_vol

                df_position_tracking.loc[df_position_tracking['TIME_OPEN_POSITION'] > idt, 'NEW_VOL'] = vol
                df_position_tracking['NEW_USED_MARGINS'] = -1 * df_position_tracking['NEW_VOL'] * np.abs(df_position_tracking['SIGNAL']) * self.margins_level * df_position_tracking['CLOSE_open'] / self.asset.lot_per_asset
                
                df_position_tracking['NEW_PNL'] = (df_position_tracking['CLOSE_close'] - df_position_tracking['CLOSE_open'])*df_position_tracking['SIGNAL']*df_position_tracking['FLAG_VALID_POSITION']*(df_position_tracking['NEW_VOL']*100)
                df_position_tracking['NEW_PNL'] = np.where(df_position_tracking['NEW_PNL'] < self.base_SL*-1*(df_position_tracking['NEW_VOL']*100), self.base_SL*-1*(df_position_tracking['NEW_VOL']*100), 
                                            np.where(df_position_tracking['NEW_PNL'] > self.base_TP*(df_position_tracking['NEW_VOL']*100), self.base_TP*(df_position_tracking['NEW_VOL']*100), 
                                                    np.where(df_position_tracking['NEW_PNL'].isnull() == True, 0, df_position_tracking['NEW_PNL'])))
                df_position_tracking['NEW_PNL'] = np.where(df_position_tracking['FLAG_VALID_POSITION'] == 1, df_position_tracking['NEW_PNL'], 0)
        

        df_position_tracking['NEW_VOL'] = np.where(df_position_tracking['NEW_VOL'].isnull(), df_position_tracking['VOL'], df_position_tracking['NEW_VOL'])
        df_position_tracking['NEW_USED_MARGINS'] = np.where(df_position_tracking['NEW_USED_MARGINS'].isnull(), df_position_tracking['USED_MARGINS'], df_position_tracking['NEW_USED_MARGINS'])
        df_position_tracking['NEW_PNL'] = np.where(df_position_tracking['NEW_PNL'] == 0, df_position_tracking['PNL'], df_position_tracking['NEW_PNL'])


        df_temp = df_position_tracking[['TIME_OPEN_POSITION', 'NEW_VOL', 'NEW_USED_MARGINS', 'POSITION_ID', 'TIME_CLOSE_POSITION', 'NEW_PNL']].copy()
        df_temp.columns = ['TIME_OPEN_POSITION', 'VOL', 'USED_MARGINS', 'POSITION_ID', 'TIME_CLOSE_POSITION', 'PNL']    

        df_balance_tracking_new = self._cal_balance(df_is = self.df_is, df_position_tracking = df_temp)
        return(df_position_tracking, df_balance_tracking_new)

    def prepare_report(self):
        df_position_tracking = self._track_positions(df_is = self.df_is,
                                        alpha = self.alpha)

        df_balance_tracking = self._cal_balance(df_is = self.df_is, 
                                        df_position_tracking = df_position_tracking)

        self.df_position_tracking = df_position_tracking
        self.df_balance_tracking = df_balance_tracking
                 
        if self.re_allocation:
            self.df_position_tracking, self.df_balance_tracking_new = self._re_balance(df_position_tracking = self.df_position_tracking, df_balance_tracking = self.df_balance_tracking)    

    def display_report(self):
        if self.df_position_tracking.empty and self.df_balance_tracking.empty:
            self.prepare_report()

        if self.re_allocation:
            size = (35, 30)
        else:
            size = (35, 20)

        fig = plt.figure(figsize = size)

        ax = fig.add_subplot(5, 1, 1)
        ax.plot(self.df_balance_tracking.loc[self.df_balance_tracking['BALANCE'] > 0, 'BALANCE'], color = 'blue', label = 'BALANCE')
        ax.plot(self.df_balance_tracking.loc[self.df_balance_tracking['BALANCE'] > 0, 'FREE_MARGINS'], color = 'green', label = 'FREE_MARGINS', alpha = 0.5)

        ax_0 = ax.twinx()
        ax_0.plot(self.df_balance_tracking.loc[self.df_balance_tracking['BALANCE'] > 0, 'DRAWDOWN'], color = 'red', alpha = 0.5, label = 'DRAWDOWN')
        ax_0.hlines(y = -0.05, xmin = self.df_balance_tracking.loc[self.df_balance_tracking['BALANCE'] > 0, :].index[0], 
                    xmax = self.df_balance_tracking.loc[self.df_balance_tracking['BALANCE'] > 0, :].index[-1], 
                    color='r', linestyles = '--', alpha = 0.3)
        
        ax.legend(bbox_to_anchor = (1.15, 1))
        ax_0.legend(bbox_to_anchor = (1.15, 0.8))
        ax.set_title('Original Balance and Drawdown (No re-allocation)')
        
        
        df_summary = pd.concat(
            [
                pd.pivot_table(
                    self.df_position_tracking[(self.df_position_tracking['FLAG_VALID_POSITION'] == 1) & (self.df_position_tracking['SIGNAL'] != 0) & (self.df_position_tracking['PNL'] > 0)],
                    index = 'SIGNAL',
                    values = 'CLOSE_open',
                    aggfunc = 'count',
                    margins = True
                ),
                pd.pivot_table(
                    self.df_position_tracking[(self.df_position_tracking['FLAG_VALID_POSITION'] == 1) & (self.df_position_tracking['SIGNAL'] != 0) & (self.df_position_tracking['PNL'] < 0)],
                    index = 'SIGNAL',
                    values = 'CLOSE_open',
                    aggfunc = 'count',
                    margins = True
                )
            ],
            axis = 1
        )

        df_summary.columns = ['WINNING_POSITIONS', 'LOSING_POSITIONS']
        df_summary.index = ['SHORT', 'LONG', 'ALL']
        df_summary = df_summary.fillna(0)
        df_summary['TOTAL_POSITIONS'] = df_summary['WINNING_POSITIONS'] + df_summary['LOSING_POSITIONS']
        
        df_summary['PERC_WINNING_POSITIONS'] = df_summary['WINNING_POSITIONS']/df_summary['TOTAL_POSITIONS']
        df_summary['PERC_LOSING_POSITIONS'] = df_summary['LOSING_POSITIONS']/df_summary['TOTAL_POSITIONS']
        df_summary['PERC_TOTAL_POSITIONS'] = df_summary['TOTAL_POSITIONS']/df_summary['TOTAL_POSITIONS']
        
        ax = fig.add_subplot(5, 1, 2)
        bars1 = ax.bar(df_summary.index, df_summary['PERC_WINNING_POSITIONS'], color = 'green', label = 'WINNING_POSITIONS')
        bars2 = ax.bar(df_summary.index, df_summary['PERC_LOSING_POSITIONS'], bottom = df_summary['PERC_WINNING_POSITIONS'], color = 'red', label = 'LOSING_POSITIONS')
        
        # Add data labels
        ax.bar_label(bars1, labels=[f'{v}' for v in df_summary['WINNING_POSITIONS']], padding = -100)
        ax.bar_label(bars2, labels=[f'{v}' for v in df_summary['LOSING_POSITIONS']], padding = -100)


        ax.legend()
        ax.set_title('Distribution of winning/ losing positions')

        ax = fig.add_subplot(5, 1, 3)
        ax.plot(self.df_balance_tracking.loc[self.df_balance_tracking['BALANCE'] > 0, 'NUM_CURRENT_POSITIONS'], color = 'blue', label = 'NUM_CURRENT_POSITIONS')
        ax.legend(bbox_to_anchor = (1.15, 1))        
        ax.set_title('Number of positions')

        if self.re_allocation:
            ax = fig.add_subplot(5, 1, 4)

            ax.plot(self.df_balance_tracking_new.loc[self.df_balance_tracking_new['BALANCE'] > 0, 'BALANCE'], color = 'blue', label = 'BALANCE')
            ax.plot(self.df_balance_tracking_new.loc[self.df_balance_tracking_new['BALANCE'] > 0, 'FREE_MARGINS'], color = 'green', label = 'FREE_MARGINS', alpha = 0.5)

            ax_0 = ax.twinx()
            ax_0.plot(self.df_balance_tracking_new.loc[self.df_balance_tracking_new['BALANCE'] > 0, 'DRAWDOWN'], color = 'red', alpha = 0.5, label = 'DRAWDOWN')
            ax_0.hlines(y = -0.05, xmin = self.df_balance_tracking_new.loc[self.df_balance_tracking_new['BALANCE'] > 0, :].index[0], 
                        xmax = self.df_balance_tracking_new.loc[self.df_balance_tracking_new['BALANCE'] > 0, :].index[-1], 
                        color='r', linestyles = '--', alpha = 0.3)
            
            ax.legend(bbox_to_anchor = (1.15, 1))
            ax_0.legend(bbox_to_anchor = (1.15, 0.8))    
            ax.set_title('New Balance and Drawdown (with re-allocation)')
          
            ax = fig.add_subplot(5, 1, 5)
            ax.plot(self.df_balance_tracking_new.loc[self.df_balance_tracking_new['BALANCE'] > 0, 'NUM_CURRENT_POSITIONS'], color = 'blue', label = 'NUM_CURRENT_POSITIONS')
            ax.legend(bbox_to_anchor = (1.15, 1))
            ax.set_ylim((0, 0.1))
            ax.set_title('Volume')


    

        plt.show()
