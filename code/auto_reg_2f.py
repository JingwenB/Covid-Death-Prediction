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
from autoReg_1f import autoReg1
import utils

# get next t days death number, starting from T, using hyperparamater K,
# note predict feature1, (y= feature1)
class autoReg2():
    def __init__(self):
        self.k = None
        self.model= None
    # fit linear auto reg, to predict next t days starting from T,
    # with hyperparamater K1,K2, where feature 2 from single model
    def fit(self,deathtraining, casetraining, K1,K2,TRAINERROR):
        
        ts_1 = deathtraining
        ts_2 = casetraining
        T = len(ts_1)
        y = ts_1[K1 - 1:T]  # size = T-K
        n = len(y)
        y = y.reshape(n, 1)
        X = np.ones((n, 1))
        for i in range(K1-1):
            x_j = ts_1[i:n + i].reshape(n, 1)
            X = np.concatenate((X, x_j), axis=1)

        for i in range(K2-1):
            x_j2 = ts_2[i:n + i].reshape(n, 1)
            X = np.concatenate((X, x_j2), axis=1)
        #print(X.shape)

        # train precess
        model = linear_model.LeastSquares()
        model.fit(X, y)
        self.model = model
        #print(model.w)
        if TRAINERROR:
            y_hat = model.predict(X)
            #print(y_hat, y)
            #plt.plot(y_hat)
            tr_error = np.mean((y-y_hat)**2)
            print("Training error     = %.3f" % tr_error)
            
        

        #return [trainError,testError, self.model, y_test_overall, y]
    
    # use existing model with selected k to run prediction for next t days, not about to test with test error
    def predict(self,deathtraining, casetraining, t,K):
        ts_1 = deathtraining
        ts_2 = casetraining
        model = self.model
        
        T = len(ts_1)
        y_pred_overall = []
        predicted_case = autoReg1()
        predicted_case.fit(casetraining, K)
        pred_case = predicted_case.predic(casetraining, K,t)
        print(pred_case)
        death = ts_1[T - K:T]
        case = ts_2[T - K:T]
        
        for i in range(t):
            
            X_pred = np.concatenate(([1], death[i+1: ], case[i+1: ]), axis=0).reshape(1, 2*K-1)
            y_pred = model.predict(X_pred).reshape(1, 1)
            y_pred_overall.append(y_pred)
            print("predicted number: %.3f,for day: %d" % ( y_pred, i))
            death = np.append(death, y_pred)
            case = np.append(case, pred_case[i])
        
        return y_pred_overall


if __name__ == "__main__":
    filename = "phase2_training_data.csv"
    with open(os.path.join("..", "data", filename), "rb") as f:
        raw = pd.read_csv(f)
    data  = raw[raw["country_id"] == "CA"]
    case_ts = data["cases"].values[0:290]
    case_ts = np.log(case_ts+1)
    death_ts = data["deaths"].values[0:290]
    
    #print(death_ts)


    
    t = 5
    
    y_pred_total = []
    y_true = data["deaths"].values[290:295]
    print(y_true)
    test_error_list = []
    
    minError = np.Inf
    K = 0
    I = 0


    for i in range(10, 260, 10):
        for k in range(3, 20, 2):
            print([i, k])
            K1 = k
            K2 = K1
    
            case_ts = data["cases"].values[i:290]
            case_ts = np.log(case_ts+1)
            death_ts = data["deaths"].values[i:290]
            
            case_death_model = autoReg2()
            case_death_model.fit(death_ts, case_ts,K1, K2, True)
            y_pred = case_death_model.predict(death_ts, case_ts, t, K1)
            test_error = np.sqrt(np.mean((y_pred - y_true)**2))
            if test_error < minError:
                K = k
                I = i 
                minError = test_error
            #test_error_list = np.append(test_error_list,test_error)
            
        
    
    print("The best selection of K and I are %d, %d, %.3f" ,K, I, minError)
        
    #print("test error is %.3f",test_error_list)
    #plt.plot(range(20, 210, 10), test_error_list[:19])
    
        
    
    case_ts2 = data["cases"].values[I:]
    case_ts2 = np.log(case_ts2+1)
    death_ts2 = data["deaths"].values[I:]
    case_death_model2 = autoReg2()
    case_death_model2.fit(death_ts2, case_ts2,K, K, True)
    y_pred = case_death_model.predict(death_ts2, case_ts2, t, K)
    te_error = np.sqrt(np.mean((y_pred - y_true)**2))
    print("test error is %.3f",te_error)
    
   

    
    
    
    
        
        
