from sklearn.datasets import make_regression
import pandas as pd

from pathlib import Path
# from tensorflow.keras.utils import 

import tensorflow as tf
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
path = Path(dir_path)
path = str(path) + "/ENB2012_data.xlsx"
print(path)


class Data:
    matrix = pd.read_excel(path)
    x = matrix[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    y = matrix[['Y1', 'Y2']]
    

    def create_data(self):
        x_train = Data.x[0:(len(Data.matrix)/2)]
        y_train = Data.y[0:(len(Data.matrix)/2)]
        x_test = Data.x[(len(Data.matrix)/2):len(Data.matrix)]
        y_test = Data.y[(len(Data.matrix)/2):len(Data.matrix)]
        return x_train, y_train, x_test, y_test
    