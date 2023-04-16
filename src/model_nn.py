import tensorflow as tf
import pandas as pd
from src.utils import rescale_data

class NNModel:

    def __init__(self):
        self._model = tf.keras.models.Sequential()
        self._model.add(tf.keras.layers.Dense(12, input_shape=(54,), activation='relu'))
        self._model.add(tf.keras.layers.Dense(12, activation='relu'))
        self._model.add(tf.keras.layers.Dense(7, activation='sigmoid'))
        self._model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def fit(self, X, y):
        y = pd.get_dummies(y)
        X = rescale_data(X)

        self._model.fit(X, y, epochs=1, batch_size=len(X))

    def predict(self, X, y):
        y = pd.get_dummies(y)
        X = rescale_data(X)
        pass

    def score(self, X, y):
        y = pd.get_dummies(y)
        X = rescale_data(X)
        return self._model.evaluate(X, y)
