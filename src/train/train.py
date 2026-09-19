import random

import keras
import numpy as np
import tensorflow as tf

from src.schema import InputTypes
from src.train.models import load_model
from src.dataset_creation.datasets import create_training_set, create_test_set

LOGIT_THRESHOLD = 0.0

POS_WEIGHT = 10.0

@keras.saving.register_keras_serializable(package="src.train.train")
class PureNegativeAccuracy(tf.keras.metrics.Metric):

    def __init__(self, threshold : float = LOGIT_THRESHOLD, name : str = "pure_negative_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true : tf.Tensor, y_pred : tf.Tensor, sample_weight=None) -> None:
        batch_size = tf.shape(y_true)[0]
        y_true_flat = tf.reshape(y_true, (batch_size, -1))
        y_pred_flat = tf.reshape(y_pred, (batch_size, -1))

        is_pure_negative = tf.reduce_all(tf.equal(y_true_flat, 0.0), axis=1)
        predicted_all_negative = tf.reduce_all(y_pred_flat <= self.threshold, axis=1)
        correct = tf.logical_and(is_pure_negative, predicted_all_negative)

        self.correct.assign_add(tf.reduce_sum(tf.cast(correct, tf.float32)))
        self.total.assign_add(tf.reduce_sum(tf.cast(is_pure_negative, tf.float32)))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self) -> None:
        self.correct.assign(0.0)
        self.total.assign(0.0)

def build_metrics() -> list[tf.keras.metrics.Metric]:
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=LOGIT_THRESHOLD),
        tf.keras.metrics.Precision(name="precision", thresholds=LOGIT_THRESHOLD),
        tf.keras.metrics.Recall(name="recall", thresholds=LOGIT_THRESHOLD),
        PureNegativeAccuracy(),
    ]

@keras.saving.register_keras_serializable(package="src.train.train")
def weighted_bce_loss(y_true : tf.Tensor, y_pred : tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=POS_WEIGHT)
    )

@keras.saving.register_keras_serializable(package="src.train.train")
def weighted_bce_loss(y_true : tf.Tensor, y_pred : tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(
        tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=POS_WEIGHT)
    )

def compile_model(model : tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer="adam",
        loss=weighted_bce_loss,
        metrics=build_metrics(),
    )

    return model

def set_global_seed(seed : int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def train(
    input_type : InputTypes,
    audio_duration : int,
    accepted_duration : float,
    n_train : int,
    batch_size : int,
    epochs : int,
    seed : int,
) -> tf.keras.Model:
    set_global_seed(seed)

    model = load_model(input_type, audio_duration, seed=seed)
    compile_model(model)

    train_set = create_training_set(input_type, audio_duration, accepted_duration, batch_size, seed).repeat()
    steps_per_epoch = n_train // batch_size

    model.fit(train_set, epochs=epochs, steps_per_epoch=steps_per_epoch)

    return model

def evaluate(
    model : tf.keras.Model,
    input_type : InputTypes,
    audio_duration : int,
    accepted_duration : float,
    n_test : int,
    batch_size : int,
    seed : int,
) -> dict:
    test_set = create_test_set(input_type, audio_duration, accepted_duration, batch_size=batch_size, seed=seed, n=n_test)

    return model.evaluate(test_set, return_dict=True)
