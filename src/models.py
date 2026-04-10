from src.support import *
from src.backtest import *

import tensorflow as tf
import keras
import keras_hub
import keras
import warnings
warnings.filterwarnings('ignore')


class DeepARCH(tf.keras.Model):

    def __init__(self, lstm_units=20):
        super().__init__()

        self.lstm = tf.keras.layers.LSTM(lstm_units)
        self.dense1 = tf.keras.layers.Dense(20, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, x, training=False):
        h = self.lstm(x, training=training)
        h = self.dense1(h)
        log_sigma2 = self.dense2(h)
        return log_sigma2

    def train_step(self, data):

        x, r_t = data

        with tf.GradientTape() as tape:
            log_sigma2 = self(x, training=True)

            sigma2 = tf.exp(log_sigma2) + 1e-6
            loss = tf.reduce_mean(log_sigma2 + (r_t**2) / sigma2)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):

        x, r_t = data

        log_sigma2 = self(x, training=False)
        sigma2 = tf.exp(log_sigma2) + 1e-6

        loss = tf.reduce_mean(log_sigma2 + (r_t**2) / sigma2)

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

class DeepRARCH(tf.keras.Model):

    def __init__(self, lstm_units=20):
        super().__init__()

        self.lstm = tf.keras.layers.LSTM(lstm_units)
        self.dense1 = tf.keras.layers.Dense(20, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)
        self.dense3 = tf.keras.layers.Dense(1, activation='softplus')
        self.log_sigma2_u = tf.Variable(0.0, trainable=True, dtype=tf.float32)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, x, rv, training=False):

        inputs = tf.concat([x, rv], axis=2)

        h = self.lstm(inputs, training=training)
        h = self.dense1(h)
        log_sigma2 = self.dense2(h)
        rv_hat = self.dense3(h)
        return log_sigma2, rv_hat, self.log_sigma2_u

    def train_step(self, data):

        (x, rv), (r_t, rv_t) = data

        with tf.GradientTape() as tape:
            log_sigma2, rv_hat, log_sigma2_u = self(x, rv, training=True)
            sigma2 = tf.exp(log_sigma2) + 1e-6
            sigma_u2 = tf.exp(log_sigma2_u)

            loss = tf.reduce_mean(
                log_sigma2
                + (r_t**2) / sigma2
                + 0.5 * log_sigma2_u
                + (rv_t - rv_hat)**2 / (2 * sigma_u2)
            )

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):

        (x, rv), (r_t, rv_t) = data
        log_sigma2, rv_hat, log_sigma2_u = self(x, rv, training=False)
        sigma2 = tf.exp(log_sigma2) + 1e-6
        sigma_u2 = tf.exp(log_sigma2_u)

        loss = tf.reduce_mean(
            log_sigma2
            + (r_t**2) / sigma2
            + 0.5 * log_sigma2_u
            + (rv_t - rv_hat)**2 / (2 * sigma_u2)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

class DeepLLMRARCH(tf.keras.Model):

    def __init__(self, lstm_units=20):
        super().__init__()

        self.lstm = tf.keras.layers.LSTM(lstm_units)

        self.dense1 = tf.keras.layers.Dense(20, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)
        self.dense3 = tf.keras.layers.Dense(1, activation='softplus')
        self.text_projection = tf.keras.layers.Dense(16, activation="relu")

        self.log_sigma2_u = tf.Variable(0.0, trainable=True, dtype=tf.float32)

        # Text encoder
        self.text_model = self._get_text_model()

        # Enable LoRA
        self.text_model.quantize("int8")
        self.text_model.backbone.enable_lora(rank=4)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def _get_text_model(self):

        text_input = keras.Input(shape=(), dtype="string")

        preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset("bert_tiny_en_uncased")
        backbone = keras_hub.models.Backbone.from_preset("bert_tiny_en_uncased")

        x = preprocessor(text_input)
        x = backbone(x)["pooled_output"]

        model = keras.Model(text_input, x)

        model.backbone = backbone

        return model

    def call(self, x, rv, text, training=False):

        text_vec = self.text_model(text, training=training)      # (batch, 768)
        text_vec = self.text_projection(text_vec)  # (batch, 32)

        ts_inputs = tf.concat([x, rv], axis=2)

        h = self.lstm(ts_inputs, training=training)

        h = tf.concat([h, text_vec], axis=1)

        h = self.dense1(h)

        log_sigma2 = self.dense2(h)
        rv_hat = self.dense3(h)

        return log_sigma2, rv_hat, self.log_sigma2_u

    def train_step(self, data):

        (x, rv, text), (r_t, rv_t) = data

        with tf.GradientTape() as tape:

            log_sigma2, rv_hat, log_sigma2_u = self(x, rv, text, training=True)

            sigma2 = tf.exp(log_sigma2) + 1e-6
            sigma_u2 = tf.exp(log_sigma2_u)

            loss = tf.reduce_mean(
                log_sigma2
                + (r_t**2) / sigma2
                + 0.5 * log_sigma2_u
                + (rv_t - rv_hat)**2 / (2 * sigma_u2)
            )

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):

        (x, rv, text), (r_t, rv_t) = data

        log_sigma2, rv_hat, log_sigma2_u = self(x, rv, text, training=False)

        sigma2 = tf.exp(log_sigma2) + 1e-6
        sigma_u2 = tf.exp(log_sigma2_u)

        loss = tf.reduce_mean(
            log_sigma2
            + (r_t**2) / sigma2
            + 0.5 * log_sigma2_u
            + (rv_t - rv_hat)**2 / (2 * sigma_u2)
        )

        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

