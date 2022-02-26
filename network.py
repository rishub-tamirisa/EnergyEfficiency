from xml.sax.xmlreader import XMLReader
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from numpy.random import rand
from numpy.random import randint
from tensorflow.keras.layers import Conv2DTranspose, LeakyReLU, BatchNormalization, Activation, Reshape, MaxPool2D
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


class Network:
    def model(numInputs, numOutputs, lr=0.0002):

        m = Sequential([
            Dense(20, input_dim=numInputs, kernel_initializer='he_uniform', activation='relu'),
            # Dense(10, input_dim=numInputs),
	        Dense(numOutputs)
        ])
        m.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr, beta_1=0.5), metrics=['accuracy'])
        return m