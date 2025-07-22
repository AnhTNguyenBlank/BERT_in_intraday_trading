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

from sklearn.model_selection import train_test_split



if __name__ == '__main__':

    news_data = pd.read_pickle("D:/BERT_in_intraday_trading/Training/Data/news_data_w_labels.pkl")

    texts = news_data.dropna()['CONTENT']
    labels = news_data.dropna()[['FLAG_HIGH_RISK', 'RATIO_MEAN_A_B', 'RATIO_VAR_A_B']]

    # Split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size = 0.7, 
        random_state = 12345, shuffle = False
        )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size = 0.5, 
        random_state = 12345, shuffle = False)

    # Wrap into datasets
    def prepare_labeled_dataset(texts, labels, batch_size = 32):
        ds = tf.data.Dataset.from_tensor_slices((texts, labels))
        ds = ds.shuffle(buffer_size=len(texts))
        ds = ds.batch(batch_size).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    train_ds = prepare_labeled_dataset(train_texts, train_labels)
    val_ds = prepare_labeled_dataset(val_texts, val_labels)
    test_ds = prepare_labeled_dataset(test_texts, test_labels)
    
    tfhub_handle_encoder = 'D:/BERT_in_intraday_trading/bert-tensorflow2-bert-en-uncased-l-10-h-128-a-2-v2'
    tfhub_handle_preprocess = 'D:/BERT_in_intraday_trading/bert-tensorflow2-en-uncased-preprocess-v3'

    bert_model = GAMMA_BERT(tfhub_handle_encoder = tfhub_handle_encoder,
                        tfhub_handle_preprocess = tfhub_handle_preprocess,
                        loss_weight = [1, 1, 1]
                        )
    print('='*100)
    print(bert_model.summary())
    print('='*100)

    # Hyperparameters
    init_lr = 3e-5
    epochs = 5
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    # Create the LR schedule
    lr_schedule = WarmUpLinearDecay(init_lr, num_warmup_steps, num_train_steps)

    # Create AdamW optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    bert_model.compile(optimizer = optimizer)
    history = bert_model.fit(train_ds, validation_data = val_ds,
                             epochs = 2)
    
    test_loss, test_binarycrossentropy, test_mse_mean, test_mse_var = bert_model.evaluate(test_ds)
    print(f'Test loss: {test_loss}')
    print(f'Test binarycrossentropy: {test_binarycrossentropy}')
    print(f'Test mse mean: {test_mse_mean}')
    print(f'Test mse var: {test_mse_var}')
    print('='*100)


    # Simulate a batch of input text (replace batch size and sequence with yours)
    dummy_input = tf.constant(["sample text here"])  # shape = (1,)

    # Call the model once to trace it
    dummy_results = bert_model(dummy_input)
    
    print(f'Dummy texts: {dummy_input}')
    print(f'Dummy results: {dummy_results}')
    print('='*100)
    
    dataset_name = 'news_data'
    saved_model_path = 'D:/BERT_in_intraday_trading/Training/Saved_results/{}_bert'.format(dataset_name.replace('/', '_'))

    print(f'Saving model to {saved_model_path}')
    bert_model.save(saved_model_path, include_optimizer=False)
    