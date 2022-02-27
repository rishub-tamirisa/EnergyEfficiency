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

input =  DecisionTreeRegressor()
ml = MultiOutputRegressor(input, n_jobs=-1)

def train_nn():
    x_train, y_train, x_test, y_test = data.Data.create_data(train_split=0.5)
    model = net.model(numInputs=8, numOutputs=2)
    model.fit(x_train, y_train, verbose=1, batch_size=20, epochs=200)
    # _, acc = ml.evaluate(x_test, y_test, verbose=1)
    y_test_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_test_pred)
    ev = explained_variance_score(y_test, y_test_pred)
    print("Neural Network MSE: %f" %(mse))
    print("Neural Network Score: %f" %(ev))
    model.save("net_classifier")
    return mse, ev

def train(iters):
    x = data.Data.x
    y = data.Data.y
    results = list()
    mean_abs_err = list()
    mean_sq_err = list()
    # define evaluation procedure
    # enumerate folds
    max = 0
    split = 0.20
    for i in range(iters):
        print("Fold: %i" %(i))
        # prepare data
        # x_train, y_train, x_test, y_test = data.Data.create_data(train_split=0.5)
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=split,random_state=2)
        ml.fit(x_train, y_train)
        # evaluate model on test set
        mae = mean_absolute_error(y_test, ml.predict(x_test))
        mse = mean_squared_error(y_test, ml.predict(x_test))
        mean_abs_err.append(mae)
        mean_sq_err.append(mse)
        acc = ml.score(x_test, y_test)
        if (acc > max):
            max = acc
            with open("model.pkl", "wb") as f:
                pickle.dump(ml, f)
        results.append(acc)
        if (split < 0.5):
            split = split + 0.1
    return results, mean_abs_err, mean_sq_err

results, mae, mse = train(iters=20)

def get_plot_metrics(iters, results, mae, mse):
    t = list(range(0,iters))
    
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.plot(t, results)
    plt.title("R^2 = " + str(round(np.mean(results), 2)))
    plt.xlabel('Fold #')
    plt.ylabel('r^2')

    plt.subplot(2, 3, 3)
    plt.plot(t, mae)
    plt.title("MAE = " + str(round(np.mean(mae), 2)))
    plt.xlabel('Fold #')
    plt.ylabel('MAE')
    
    plt.subplot(2, 3, 5)
    plt.plot(t, mse)
    plt.title("MSE = " + str(round(np.mean(mse), 2)))
    plt.xlabel('Fold #')
    plt.ylabel('MSE')

    plt.savefig("model_eval")


# print('Score: %.3f (%.3f)' % (np.mean(results), np.std(results)))
# print('EVS: %.3f (%.3f)' % (np.mean(mae), np.std(mae)))
# print("==============================")

get_plot_metrics(20, results, mae, mse)

# train_nn()

