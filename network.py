import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from numpy.random import rand
from numpy.random import randint
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


class Network:
    def model(numInputs, numOutputs, lr=0.0002):

        m = Sequential([
            Dense(100, input_dim=numInputs, kernel_initializer='he_uniform', activation='relu'),
            Dense(100, activation="relu"),
            Dense(50, activation="relu"),
	        Dense(numOutputs)
        ])
        m.compile(loss='mse', optimizer=Adam(learning_rate=lr, beta_1=0.5), metrics=['accuracy'])
        return m