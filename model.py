from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import network
import data
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors, RadiusNeighborsRegressor
import pickle
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import matplotlib.pyplot as plt

import numpy as np
net = network.Network

# model = network.Network.model(numInputs=8, numOutputs=2)
# model.build()
# print(model.summary())
knn =  DecisionTreeRegressor()#KNeighborsRegressor(n_neighbors=1)
ml = MultiOutputRegressor(knn, n_jobs=-1)

def train_nn():
    x_train, y_train, x_test, y_test = data.Data.create_data()
    model = net.model(numInputs=8, numOutputs=2)
    model.fit(x_train, y_train, verbose=1, batch_size=20, epochs=200)
    # _, acc = ml.evaluate(x_test, y_test, verbose=1)
    y_test_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_test_pred)
    ev = explained_variance_score(y_test, y_test_pred)
    print("Neural Network MSE: %f" %(mse))
    print("Neural Network Score: %f" %(ev))
    model.save("net_classifier")

def train():
    x = data.Data.x
    y = data.Data.y
    results = list()
    mean_abs_err = list()
    # define evaluation procedure
    # enumerate folds
    max = 0
    for i in range(10):
        print("Fold: %i" %(i))
        # prepare data
        # x_train, y_train, x_test, y_test = data.Data.create_data()
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.50,random_state=2)

        #build model
        # model = net.model(numInputs=8, numOutputs=2)
        # fit model
        # print("==================TRAINING=====================")
        # model.fit(x_train, y_train, verbose=1, epochs=50)
        # print(x_train)
        # print(y_train)
        ml.fit(x_train, y_train)

        # evaluate model on test set
        # print("===================TESTING=====================")
        # _, acc = ml.evaluate(x_test, y_test, verbose=1)
        mae = mean_squared_log_error(y_test, ml.predict(x_test))
        mean_abs_err.append(mae)
        acc = ml.score(x_test, y_test)
        # store result
        #print(model.metrics_names)
        # print(acc)
        if (acc > max):
            max = acc
            # ml.save("classifier")
            with open("model.pkl", "wb") as f:
                pickle.dump(ml, f)
        results.append(acc)
    # model.save("classifier")
    return results, mean_abs_err


results, mae = train()
# print(results)
print('Score: %.3f (%.3f)' % (np.mean(results), np.std(results)))
print('EVS: %.3f (%.3f)' % (np.mean(mae), np.std(mae)))
print("==============================")
train_nn()