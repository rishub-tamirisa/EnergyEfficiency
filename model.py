from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import network
import data
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors, RadiusNeighborsRegressor
import pickle

import numpy as np
net = network.Network

# model = network.Network.model(numInputs=8, numOutputs=2)
# model.build()
# print(model.summary())
knn =  DecisionTreeRegressor()#KNeighborsRegressor(n_neighbors=1)
ml = MultiOutputRegressor(knn, n_jobs=-1)



def train():
    x = data.Data.x
    y = data.Data.y
    results = list()
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # enumerate folds
    max = 0
    # for train_ix, test_ix in cv.split(x):
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
    acc = ml.score(x_test, y_test)
    # store result
    #print(model.metrics_names)
    # print(acc)
    # if (acc > max):
    #     max = acc
    # ml.save("classifier")
    with open("model.pkl", "wb") as f:
        pickle.dump(ml, f)
    results.append(acc)
    # model.save("classifier")
    return results


results = train()
# print(results)
print('Score: %.3f (%.3f)' % (np.mean(results), np.std(results)))