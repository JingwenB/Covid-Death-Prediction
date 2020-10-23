# basics
import argparse
import os
import pickle

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    filename = "phase1_training_data.csv"
    with open(os.path.join("..", "data", filename), "rb") as f:
        raw = pd.read_csv(f)
    ca = raw[raw["country_id"]=="CA"]

    # ad_cases = ad["cases"].values
    total_days = 280
    ca_14_cases = ca[["date","cases_14_100k"]].values
    ca_100_cases = ca[["date", "cases_100k"]].values
    ca_deaths = ca[["date","deaths"]].values
    ca_deaths_all = ca_deaths[:,1].reshape(total_days,1)

    ca_deaths_off = np.zeros(14).reshape(14,1)
    sth = ca_deaths_all[:total_days-14,].reshape(total_days-14,1)
    ca_deaths_off = np.concatenate((ca_deaths_off,sth),axis=0)
    ca_deaths_14 = np.subtract(ca_deaths_all,ca_deaths_off)

    # a = np.subtract([[1],[2]], [[0], [2]])
    # b = np.array([[5, 6]])
    # a =np.concatenate((a, b), axis=0)


    us = raw[raw["country_id"] == "US"]
    us_cases = us[["date", "cases_14_100k"]].values
    us_deaths = us[["date", "deaths"]].values

    unique = raw["country_id"].unique()

    selected_country = raw[raw["country_id"] == "CA"][["date", "deaths", "cases", "cases_100k", "cases_14_100k"]]


    # ca = raw[raw["country_id"] == "CA"][["date", "deaths", "cases", "cases_100k", "cases_14_100k"]].values
    # us = raw[raw["country_id"] == "US"][["date", "deaths", "cases", "cases_100k", "cases_14_100k"]].values
    # ca = np.concatenate((ca, us),axis=1)

    # for x in unique:
    #     if x != "CA" and x != "nan":
    #         print(x)
    #         curr = raw[raw["country_id"]==x][["deaths","cases","cases_100k", "cases_14_100k"]]
    #         selected_country = np.concatenate((selected_country, curr),axis=1)

    # df = pd.DataFrame(selected_country, columns=["date", "deaths_ca", "cases_ca", "cases_100k_ca", "cases_14_100k_ca"
    #                                              , "deaths_uk", "cases_uk", "cases_100k_uk", "cases_14_100k_uk"
    #                                              , "deaths_us", "cases_us", "cases_100k_us", "cases_14_100k_us"])
    #
    # df.to_csv(os.path.join("..", "data", "out.csv"))


    # st =[1, 2,3]
    # st2 = [1, 2, 3]
    # fig = plt.figure()
    # fig.xcorr(st,st2)
    # plt.figure()
    # plt.plot([1, 2, 3, 4], 'r--', [1, 4, 9, 16], 'ro')
    # plt.show()

    start_k = 100
    plt.figure()
    N, D = ca_14_cases.shape
    a = ca_14_cases[start_k:, 1]
    plt.plot(a, 'bs')
    filename = os.path.join("..", "figs", "date_14_cases")
    print("Saving", filename)
    plt.savefig(filename)
    plt.clf()

    plt.figure()
    a = ca_100_cases[start_k:, 1]
    plt.plot(a, 'bs')
    filename = os.path.join("..", "figs", "date_100_cases")
    print("Saving", filename)
    plt.savefig(filename)
    plt.clf()

    plt.figure()
    a = ca_deaths[start_k:, 1]
    plt.plot(a, 'bs')
    filename = os.path.join("..", "figs", "date_100_death")
    print("Saving", filename)
    plt.savefig(filename)
    plt.clf()

    plt.figure()
    b = ca_deaths_14[start_k:, ]
    plt.plot(b, 'bs')
    filename = os.path.join("..", "figs", "date_14_death")
    print("Saving", filename)
    plt.savefig(filename)
    plt.clf()

    plt.figure()
    # x ca_cases y ca_deaths
    plt.plot(ca_14_cases[start_k:, 1], ca_deaths_14[start_k:,], 'ro')
    filename = os.path.join("..", "figs", "14_case_death")
    print("Saving", filename)
    plt.savefig(filename)
    plt.clf()

    plt.figure()
    # x ca_cases y ca_deaths
    plt.plot(ca_100_cases[start_k:, 1], ca_deaths[start_k:, 1], 'ro')
    filename = os.path.join("..", "figs", "100_case_death")
    print("Saving", filename)
    plt.savefig(filename)
    plt.clf()

    plt.figure()
    # x ca_cases y ca_deaths
    plt.plot(us_cases[start_k:, 1], us_deaths[start_k:, 1], 'ro')
    filename = os.path.join("..", "figs", "us_date_case_death")
    print("Saving", filename)
    plt.savefig(filename)

    plt.clf()

    # plt.figure()
    # # x ca_cases y ca_deaths
    # plt.plot( ca_deaths[200:, 1],ca_14_cases[200:, 1], 'ro')
    # filename = os.path.join("..", "figs", "death_case")
    # print("Saving", filename)
    # plt.savefig(filename)


    # filename = os.path.join("..", "figs", "case_date_death")
    # print("Saving", filename)
    # plt.savefig(filename)
    # plt.show()


    x = np.array([11.37, 14.23, 16.3, 12.36,
         6.54, 4.23, 19.11, 12.13,
         19.91, 11.00])

    y = np.array([15.21, 12.23, 4.76, 9.89,
         8.96, 19.26, 12.24, 11.54,
         13.39, 18.96])

    # Plot graph
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    #
    # # cross correlation using
    # # xcorr() function
    # ax1.xcorr(x, y, usevlines=True,
    #           maxlags=9, normed=True,
    #           lw=2)
    # # adding grid to the graph
    # ax1.grid(True)
    # ax1.axhline(0, color='blue', lw=2)
    #
    # # show final plotted graph
    # plt.show()

    # print(len(sth))