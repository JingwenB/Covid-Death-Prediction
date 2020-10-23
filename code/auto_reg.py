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

# get next  days case number, starting from T, using hyperparamater K
def get_next_case_num(raw, country,t,T,K):
    print(" " )

    # print("predicting for country:%s" % (country))

    ca = raw[raw["country_id"] == country]
    ca_case_ts = ca["cases_100k"].values  # 280*1

    y = ca_case_ts[K - 1:T]  # size = T-K
    n = len(y)
    y = y.reshape(n, 1)
    X = np.ones((n, 1))
    for i in range(K - 1):
        x_j = ca_case_ts[i:n + i].reshape(n, 1)
        X = np.concatenate((X, x_j), axis=1)

    # train
    model = linear_model.LeastSquares()
    model.fit(X, y)
    # print(model.w)

    yhat = model.predict(X)
    trainError = np.mean((yhat - y) ** 2)
    print("Training error = %.3f" % trainError)

    # T+1
    # x_T+1 = [1, dT−K+1, dT −K+2, . . . , dT]
    X_test = np.concatenate(([1], ca_case_ts[T - K + 1:T]), axis=0).reshape(1, K)
    y_test = np.array(ca_case_ts[T]).reshape(1, 1)
    y_pred_curr = model.predict(X_test)
    # print("predicted case number: %.3f, real case number: %.3f" % (y_pred_curr, y_test))

    # predict
    for i in range(1, t):
        # print("day#: %d" % (i + 1))
        X_test_i = np.concatenate(([1], ca_case_ts[T - K + 1 + i:T + i]), axis=0).reshape(1, K)
        y_test_i = np.array(ca_case_ts[T + i]).reshape(1, 1)
        y_pred_curr = model.predict(X_test_i)
        # print("predicted case number: %.3f, real case number: %.3f" % (y_pred_curr, y_test_i))

        X_test = np.concatenate((X_test, X_test_i), axis=0)
        y_test = np.concatenate((y_test, y_test_i), axis=0)

    y_pred = model.predict(X_test)
    testError = np.mean((y_pred - y_test) ** 2)
    print("Test error     = %.3f" % testError)
    return [trainError,testError]

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    filename = "phase1_training_data.csv"
    with open(os.path.join("..", "data", filename), "rb") as f:
        raw = pd.read_csv(f)

    tr = []
    te = []

    for k in range(3, 50):
        retval = get_next_case_num(raw, "CA", 5, 270, k)
        tr.append(retval[0])
        te.append(retval[1])

    te_min = np.min(te)
    min_k = np.argmin(te)+3

    print("min: %.5f"% te_min)
    plt.figure()
    plt.plot(te, "r", tr, "b")
    filename = os.path.join("..", "figs", "auto_reg_error.png")
    print("Saving", filename)
    plt.savefig(filename)
    plt.clf()

    get_next_case_num(raw, "CA", 5, 270, min_k)




