from tensorflow import keras
import data
import pandas as pd
import numpy as np
import pickle

x = data.Data.x
y = data.Data.y

def input (x1, x2, x3, x4, x5, x6, x7, x8):
    dict = {'X1':[x1],
            'X2':[x2],
            'X3':[x3],
            'X4':[x4],
            'X5':[x5],
            'X6':[x6],
            'X7':[x7],
            'X8':[x8],
        }
    dict = pd.DataFrame(dict)
    return dict

#true input
def get_prediction(x1, x2, x3, x4, x5, x6, x7, x8) :
    user_input = input(x1,x2,x3,x4,x5,x6,x7,x8)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        output = model.predict(dict)

        # print(model.predict(user_input))
        return output[0][0], output[0][1]

# testing purposes
def get_prediction_test(dict: pd.DataFrame, i) :
    print("Actual Values: " + str(y.iloc[[i]]))
    ml = keras.models.load_model('net_classifier')
    print("Neural Network Prediction: " + str(ml.predict(dict)))
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        output = model.predict(dict)

        print("Decision Tree Prediction: " + str(output))


# print(get_prediction_test(x.iloc[[740]], i = 740))

