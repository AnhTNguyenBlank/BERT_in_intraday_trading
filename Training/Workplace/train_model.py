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

    with open("D:/BERT_in_intraday_trading/Training/Data/stored_data.pkl", "rb") as f:
        news_data = pickle.load(f)

    pass