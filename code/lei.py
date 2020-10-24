# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np  # this comes with Anaconda
import pandas as pd  # this comes with Anaconda
import matplotlib.pyplot as plt  # this comes with Anaconda
from scipy.optimize import approx_fprime  # this comes with Anaconda
from sklearn.tree import DecisionTreeClassifier  # if using Anaconda, install with `conda install scikit-learn`
import linear_model




""" NOTE:
Python is nice, but it's not perfect. One horrible thing about Python is that a 
package might use different names for installation and importing. For example, 
seeing code with `import sklearn` you might sensibly try to install the package 
with `conda install sklearn` or `pip install sklearn`. But, in fact, the actual 
way to install it is `conda install scikit-learn` or `pip install scikit-learn`.
Wouldn't it be lovely if the same name was used in both places, instead of 
`sklearn` and then `scikit-learn`? Please be aware of this annoying feature. 
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        print("question 1:")
        y_accurate = np.array((35, 10, 27, 42, 0, 0, 28, 16, 11, 26, 23)).reshape(11, 1)
        # print(y_accurate.shape)
        data = pd.read_csv(os.path.join('..', 'data', 'phase1_training_data.csv'))
        data_values = data.values
        ca_data_values = data[data["country_id"] == "CA"]

        ca_deaths_values = ca_data_values["deaths"].values
        print(ca_deaths_values)
        total_dates = ca_deaths_values.shape[0]
        for k in range(14):
            print("k = ", k)
            ca_deaths_values_t_interval = ca_data_values["deaths"].values[20 * k:]
            t = np.arange(total_dates)[20 * k:]
            t_length = t.size
            X = np.transpose(t).reshape(t_length, 1)
            # print(X.shape)
            y = ca_deaths_values_t_interval.reshape(t_length, 1)
            # print(y.shape)

            predict_t_interval = np.arange(280, 291).reshape(11, 1)
            print(predict_t_interval)

            print(predict_t_interval.shape)
            # plt.plot(t, ca_deaths_values_t_interval)
            # plt.show()


            # for p in range(10):
            #     print("p = ", p)
            #     model = linear_model.LeastSquaresPoly(p)
            #     model.fit(X, y)
            #     plt.plot(X, y)
            #     y_model = model.predict(X)
            #     plt.plot(X, y_model)
            #     plt.show()
            #     y_pred = model.predict(predict_t_interval)
            #     # print(y_pred)


            # for p in range(1, 10):
            #     for k in range(1, 10):
            #         print("p = ", p)
            #         print("k = ", k)
            #         model = linear_model.LeastSquaresPolySin(p, k)
            #         model.fit(X, y)
            #         plt.plot(X, y)
            #         y_model = model.predict(X)
            #         plt.plot(X, y_model)
            #         plt.show()
            #         y_pred = model.predict(predict_t_interval)

            for p in range(3, 10):
                for k in range(1, 10):
                    print("p = ", p)
                    print("k = ", k)
                    model = linear_model.LeastSquaresPolySinMulti(p, k)
                    model.fit(X, y)
                    plt.plot(X, y)
                    y_model = model.predict(X)
                    plt.plot(X, y_model)
                    plt.show()
                    y_pred = model.predict(predict_t_interval)
                    print(y_pred)

    else:
        print("No code to run for question", question)