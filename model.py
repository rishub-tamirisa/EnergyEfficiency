from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import network
import data
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors, RadiusNeighborsRegressor
import pickle
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


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
    mean_rsq_err = list()
    mean_sq_err = list()
    mean_sq_err_diff = list()
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    # define evaluation procedure
    # enumerate folds
    max = 0
    split = 0.5
    for i in range(iters):
        print("Fold: %i" %(i))
        # prepare data
        # x_train, y_train, x_test, y_test = data.Data.create_data(train_split=0.5)
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=split,random_state=np.random)
        ml.fit(x_train, y_train)
        # evaluate model on test set
        prediction = ml.predict(x_test)
        y_test_diffs = []
        predict_diffs = []
        for i in range(len(prediction)):
            diff = prediction[i][0] - prediction[i][1]
            y_diff = y_test.iloc[[i]]['Y1'] - y_test.iloc[[i]]['Y2']
            y_diff = np.asarray(y_diff)[0]
            # print(diff)
            # print(y_diff)

            y_test_diffs.append(y_diff)
            predict_diffs.append(diff)

        mse_diff = mean_squared_error(y_test_diffs, predict_diffs, squared=False)
        rmse = mean_squared_error(y_test, ml.predict(x_test), squared=False)
        mse = mean_squared_error(y_test, ml.predict(x_test), squared=True)

        mean_sq_err_diff.append(mse_diff)
        mean_rsq_err.append(rmse)
        mean_sq_err.append(mse)
        acc = ml.score(x_test, y_test)
        if (acc > max):
            max = acc
            with open("model.pkl", "wb") as f:
                pickle.dump(ml, f)
        results.append(acc)
        # if (split < 0.5):
        #     split = split + 0.1

        
        # print(cvs)
    cvs = cross_val_score(ml, x, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    cvs = np.sqrt(np.square(cvs))
    return results, mean_rsq_err, mean_sq_err, mean_sq_err_diff, cvs

results, rmse, mse, mse_diff, cvs = train(iters=50)

def get_plot_metrics(iters, results, rmse, mse, mse_diff, cvs):
    t = list(range(0,iters))
    tc = list(range(10))
    
    plt.figure()
    plt.subplot(3, 3, 1)
    plt.plot(t, results)
    plt.title("Norm R^2 = " + str(round(np.mean(results), 2)))
    plt.xlabel('Fold #')
    plt.ylabel('r^2')
    plt.ylim(0.8, 1.0)
    
    plt.subplot(3, 3, 3)
    plt.plot(t, rmse)
    plt.title("Norm RMSE = " + str(round(np.mean(rmse), 2)) + "±" + str(round(np.std(rmse), 2)))
    plt.xlabel('Fold #')
    plt.ylabel('RMSE') #interpretable in same units as cooling load and heating load
    plt.ylim(1.0, 5.0)
    
    print(cvs)
    plt.subplot(3, 3, 5)
    plt.plot(tc, cvs)
    plt.title("CV MAE = " + str(round(np.mean(cvs), 2)) + "±" + str(round(np.std(cvs), 2)))
    plt.xlabel('Fold #')
    plt.ylabel('MAE') 
    # plt.ylim(1.0, 5.0)

    plt.subplot(3, 3, 7)
    plt.plot(t, mse)
    plt.title("Norm MSE = " + str(round(np.mean(mse), 2)) + "±" + str(round(np.std(mse), 2)))
    plt.xlabel('Fold #')
    plt.ylabel('MSE')
    plt.ylim(1.0, 5.0)

    plt.subplot(3, 3, 9)
    plt.plot(t, mse_diff)
    plt.title("Diff MSE = " + str(round(np.mean(mse_diff), 2)) + "±" + str(round(np.std(mse_diff), 2)))
    plt.xlabel('Fold #')
    plt.ylabel('MSE')
    plt.ylim(1.0, 5.0)

    plt.savefig("model_eval")


print('Score: %.3f (%.3f)' % (np.mean(results), np.std(results)))
print('EVS: %.3f (%.3f)' % (np.mean(rmse), np.std(rmse)))
print("==============================")

get_plot_metrics(50, results, rmse, mse, mse_diff, cvs)

# train_nn()

