# basics
import argparse
import os
import pickle

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# sklearn imports
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

# get next t days death number, starting from T, using hyperparamater K
class autoReg():
    def __init__(self,raw, feature,country):
        self.raw = raw
        self.feature = feature
        self.country = country
        self.ts = raw[raw["country_id"] == country][feature].values
        self.k = None
        self.model= None

    # fit linear auto reg, to predict next t days starting from T,
    # with hyperparamater K
    def fit(self,t,T,K):

        ts = self.ts
        y = ts[K - 1:T]  # size = T-K
        n = len(y)
        y = y.reshape(n, 1)
        X = np.ones((n, 1))
        for i in range(K - 1):
            x_j = ts[i:n + i].reshape(n, 1)
            X = np.concatenate((X, x_j), axis=1)

        # train
        model = linear_model.LeastSquares()
        model.fit(X, y)

        yhat = model.predict(X)
        trainError = np.mean((yhat - y) ** 2)
        # print("Training error = %.3f" % trainError)

        # T+1
        X_test = np.concatenate(([1], ts[T - K + 1:T]), axis=0).reshape(1, K)
        y_test = np.array(ts[T]).reshape(1, 1)
        y_test_overall = y_test
        y_pred = model.predict(X_test).reshape(1, 1)
        # print("predicted case number: %.3f, real case number: %.3f" % (y_pred, y_test))
        y_pred_overall = y_pred
        # predict
        for i in range(1, t):
            # print("day#: %d" % (i + 1))
            a=X_test[:,2:]
            X_test = np.concatenate(([[1]], a, y_pred), axis=1)
            y_test_i = np.array(ts[T + i]).reshape(1, 1)
            y_test_overall = np.concatenate((y_test_overall, y_test_i), axis=0)
            y_pred = model.predict(X_test).reshape(1, 1)
            # print("predicted case number: %.3f, real case number: %.3f" % (y_pred, y_test_i))
            y_pred_overall = np.concatenate((y_pred_overall,y_pred), axis=0)

        testError = np.mean((y_pred_overall - y_test_overall) ** 2)
        # print("Test error     = %.3f" % testError)
        return [trainError,testError, model, y_test_overall, y]

    # iterate through k , to find best k with min test error
    def find_best_k(self,t,T):
        tr = []
        te = []
        k_largest= 60
        if(T<k_largest):
            k_largest = 20


        for k in range(2, k_largest):
            retval = auto_1.fit(t,T,k)
            tr.append(retval[0])
            te.append(retval[1])

        te_min = np.min(te)
        min_k = np.argmin(te) + 2

        print("min: %.5f, K:%d" % (te_min, np.argmin(te) + 2))

        self.model = auto_1.fit(t, T, min_k)[2]
        y_pred_overall =auto_1.fit(t, T, min_k)[3]
        y_test_overall = auto_1.fit(t, T, min_k)[4]
        self.K = min_k

    # use existing model with selected k to run prediction for next t days, not about to test with test error
    def predict(self,t):
        model = self.model
        ts = self.ts
        K = self.K
        T = len(ts)

        X_pred = np.concatenate(([1], ts[T - K + 1:T]), axis=0).reshape(1, K)
        y_pred = model.predict(X_pred).reshape(1, 1)
        print("predicted number: %d, for day: %d" % (y_pred, T))
        y_pred_overall = y_pred
        # predict
        for i in range(1, t):
            a = X_pred[:, 2:]
            X_pred = np.concatenate(([[1]], a, y_pred), axis=1)
            y_pred = model.predict(X_pred).reshape(1, 1)
            print("predicted number: %d,for day: %d" % ( y_pred, T+i))
            y_pred_overall = np.concatenate((y_pred_overall, y_pred), axis=0)

        return y_pred_overall


if __name__ == "__main__":
    filename = "phase1_training_data.csv"
    with open(os.path.join("..", "data", filename), "rb") as f:
        raw = pd.read_csv(f)



# train stage
    auto_1 = autoReg(raw, "deaths", "CA")
    auto_1.find_best_k(11, 268)
#  predict
    pred = auto_1.predict(11)
# write to csv
    df = pd.DataFrame(pred, columns=["death"])
    id_11 = [0,1,2,3,4,5,6,7,8,9,10]
    df.insert(1, "id", id_11, True)

    df.to_csv(os.path.join("..", "data", "phase1_prediction.csv"), index=False)


    # for i in range(1, 25):
    #     i = 10*i
    #     auto_1 = autoReg(raw, "deaths", "CA",i)
    #     auto_1.find_best_k(11, 265-i)
    #     pred = auto_1.predict(11,i)




