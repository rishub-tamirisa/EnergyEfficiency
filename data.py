from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

from pathlib import Path
# from tensorflow.keras.utils import 

# import tensorflow as tf
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
path = Path(dir_path)
path = str(path) + "/ENB2012_data.xlsx"
print(path)


class Data:
    matrix = pd.read_excel(path)
    x = matrix[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
    y = matrix[['Y1', 'Y2']]
    

    def create_data(train_split):
        div = 1 / train_split
        x_train = Data.x[0:(int(len(Data.matrix)/div))]
        y_train = Data.y[0:(int(len(Data.matrix)/div))]
        x_test = Data.x[int(len(Data.matrix)/div):int(len(Data.matrix))]
        y_test = Data.y[int(len(Data.matrix)/div):int(len(Data.matrix))]
        y_diff = y_test.iloc[[1]]['Y1'] - y_test.iloc[[1]]['Y2']
        # print(np.asarray(y_diff))
        # print(x_train)
        # print(x_test)
        return x_train, y_train, x_test, y_test
    
Data.create_data(train_split=0.5)