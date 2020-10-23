
# basics
import argparse
import os
import pickle
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

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        # item_total_rating = np.zeros(len(item_mapper))
        # for i in range(1, len(item_mapper)+2):
        #      index = item_mapper[ratings["item"][i]]
        #      item_total_rating[index] += ratings["rating"][i]
        # max_rated_item = item_inverse_mapper[np.argmax(item_total_rating)]

        total_ratings = np.sum(X_binary, axis=0)
        most_reviewed_items = item_inverse_mapper[np.argmax(total_ratings)]

        # YOUR CODE HERE FOR Q1.1.1
        print("most rated item: ",most_reviewed_items)

        # YOUR CODE HERE FOR Q1.1.2

        # user_list = ratings[ratings['item'] == most_reviewed_items]["user"]
        # user_info = []
        # for user in user_list:
        #     num_rated_item = len(ratings[ratings['user'] == user])
        #     # print("the users who rated that item: ", user)
        #     # print("# of items this user has reviewed: ", num_rated_item)
        #     user_info.append([user, num_rated_item])
        # print(user_info)

        user_ratings = np.sum(X_binary, axis=1)
        user_reviewed_most = user_inverse_mapper[np.argmax(user_ratings)]
        print("the user who reviewed most: %s, number of items reviewed: %1d" %( user_reviewed_most,np.max(user_ratings)))


        # YOUR CODE HERE FOR Q1.1.3

        # print("Number of non-zero ratings: ", ratings_num)
        print("Number of non-zero ratings for each item: ", X.getnnz(axis=0))
        item_rating_num = X.getnnz(axis=0)
        print("Number of non-zero ratings for each user: ", X.getnnz(axis=1))
        user_rating_num = X.getnnz(axis=1)

        plt.figure()
        plt.hist(item_rating_num)
        plt.yscale('log', nonposy='clip')
        # plt.legend()
        fname = os.path.join("..", "figs", "q1_item.pdf")
        plt.savefig(fname)

        plt.figure()
        plt.hist(user_rating_num)
        plt.yscale('log', nonposy='clip')
        # plt.legend()
        fname = os.path.join("..", "figs", "q1_user.pdf")
        plt.savefig(fname)

        plt.figure()
        plt.hist(ratings["rating"])
        # plt.legend()
        fname = os.path.join("..", "figs", "q1_ratings.pdf")
        plt.savefig(fname)



    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind]

        print(url_amazon % grill_brush)

        new_x = X

        # YOUR CODE HERE FOR Q1.2
        #  Euclidean distance
        neigh = NearestNeighbors(metric='minkowski',  p=2).fit(np.transpose(X))
        q1_2_1 = neigh.kneighbors(np.transpose(grill_brush_vec), n_neighbors=6, return_distance=False)
        q1_2_1 = list(map(lambda item: item_inverse_mapper[item], q1_2_1[0]))
        print("Euclidean distance", q1_2_1)

        neigh = NearestNeighbors(metric='minkowski', p=2).fit(np.transpose(normalize(X)))
        q1_2_2 = neigh.kneighbors(np.transpose(grill_brush_vec), n_neighbors=6, return_distance=False)
        q1_2_2 = list(map(lambda item: item_inverse_mapper[item], q1_2_2[0]))
        print("Normalized Euclidean distance", q1_2_2)

        neigh = NearestNeighbors(metric='cosine', p=2).fit(np.transpose(X))
        q1_2_3 = neigh.kneighbors(np.transpose(grill_brush_vec), n_neighbors=6, return_distance=False)
        item_list = list(map(lambda item: item_inverse_mapper[item], q1_2_3[0]))
        print("cosine distance", item_list)

        # YOUR CODE HERE FOR Q1.3

        # of reviews of q1_2_3
        print("popular items from Euclidean distance")
        for ind in q1_2_1[0]:
            item = item_inverse_mapper[ind]
            print("item:%s, # of reviews:  %1d" % (item, len(X[:, ind].data)))
        print("popular items from cosine distance")

        for ind in q1_2_3[0]:
            item = item_inverse_mapper[ind]
            print("item:%s, # of reviews:  %1d" % (item, len(X[:, ind].data)))



    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        mylist = []
        for i in range(0,400):
            mylist.append(1)
        for i in range(0, 100):
            mylist.append(0.1)

        z = np.array(mylist)
        np.reshape(z, (500,1))

        # Fit weighted-least-squares estimator
        ls_model = linear_model.LeastSquares()
        model = linear_model.WeightedLeastSquares()
        model.fit(X, y, z)
        print(model.w)

        utils.test_and_plot(model, X, y, title="Weighted Least Squares", filename="q3_weighted_least_squares.pdf")


    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares bias model
        model = linear_model.LeastSquaresBias()
        model.fit(X, y)

        utils.test_and_plot(model, X, y, Xtest,ytest, title="Least Squares with bias variable", filename="q4_least_squares_bias.pdf")


    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(0,11):
            print("p=%d" % p)
            # Fit least-squares model
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X, y)

            utils.test_and_plot(model, X, y, Xtest, ytest, title="%dth Polynomial Basis" % p,
                                filename="q4_poly_bias_%d.pdf"% p)

    else:
        print("Unknown question: %s" % question)

