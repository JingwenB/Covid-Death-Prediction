import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime

import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T @ X, X.T @y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares):  # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        Z = np.diag(z)
        self.w = solve(X.T@ Z @ X, X.T @  Z @  y)


# class LinearModelGradient(LeastSquares):
#
#     def fit(self,X,y):
#         n, d = X.shape
#
#         # Initial guess
#         self.w = np.zeros(d)
#
#         # check the gradient
#         estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
#         implemented_gradient = self.funObj(self.w,X,y)[1]
#         if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
#             print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
#         else:
#             print('User and numerical derivatives agree.')
#
#         self.w, f = findMin(self.funObj, self.w, 100, X, y)
#
#     def funObj(self,w,X,y):
#
#         f = 0
#         g = 0
#         for i in range(len(y)):
#             expi = np.exp(X[i] @ w - y[i])
#             logi = np.log(expi + 1 / expi)
#             f = f + logi
#             g = g + X[i] * (expi ** 2 - 1) / (expi ** 2 + 1)
#         return f, g

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE '''
        # Calculate the function value
        f = np.sum(np.log(np.exp(X @ w - y) + np.exp(y - X @ w)))

        # Calculate the gradient value
        g = X.T @ ((np.exp(X @ w - y) - np.exp(y - X @ w)) / (np.exp(X @ w - y) + np.exp(y - X @ w)))

        return (f,g)


# Least Squares with a bias added
class LeastSquaresBias:
    def fit(self,X,y):
        Z = np.ones((X.shape[0],1))
        Z = np.append(Z, X, axis =1)
        self.w = solve(Z.T@Z, Z.T@ y)

    def predict(self, X):
        Z = np.ones((X.shape[0], 1))
        Z = np.append(Z, X, axis=1)
        return Z @ self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        Z = self.__polyBasis(X)
        self.w = solve(Z.T @ Z, Z.T @ y)

    def predict(self, X):
        Z = self.__polyBasis(X)
        return Z @ self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        Z = np.ones((X.shape[0], 1))
        for i in range(1, self.p+1):
            temp_X = np.power(X, i)
            Z = np.append(Z, temp_X, axis=1)

        return Z



# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        Z = self.__polyBasis(X)
        self.w = solve(Z.T @ Z, Z.T @ y)

    def predict(self, X):
        Z = self.__polyBasis(X)
        return Z @ self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        Z = np.ones((X.shape[0], 1))
        for i in range(1, self.p+1):
            temp_X = np.power(X, i)
            Z = np.append(Z, temp_X, axis=1)
        return Z



# Least Squares with polynomial and sin basis
class LeastSquaresPolySin:
    def __init__(self, p, k):
        self.leastSquares = LeastSquares()
        self.p = p
        self.k = k

    def fit(self,X,y):
        Z = self.__polySinBasis(X)
        self.w = solve(Z.T @ Z, Z.T @ y)

    def predict(self, X):
        Z = self.__polySinBasis(X)
        return Z @ self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polySinBasis(self, X):
        # poly part
        Z = np.ones((X.shape[0], 1))
        for i in range(1, self.p+1):
            temp_X = np.power(X, i)
            Z = np.append(Z, temp_X, axis=1)

        # sin part
        sin_k = np.sin(self.k * X)
        Z = np.append(Z, sin_k, axis=1)

        return Z

# Least Squares with exp basis
class LeastSquaresExp:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self, X, y):
        n, d = X.shape
        Z = self.__expBasis(X)
        self.leastSquares.fit(Z, y)

    def predict(self, X):
        Z = self.__expBasis(X)
        return self.leastSquares.predict(Z)

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __expBasis(self, X):
        return np.exp(X * np.arange(self.p + 1))



# Least Squares with ploy basis and exp basis
class LeastSquaresPolyExp:
    def __init__(self, p, k):
        self.leastSquares = LeastSquares()
        self.p = p
        self.k = k

    def fit(self, X, y):
        n, d = X.shape
        Z = self.__polyexpBasis(X)
        self.leastSquares.fit(Z, y)

    def predict(self, X):
        Z = self.__polyexpBasis(X)
        return self.leastSquares.predict(Z)

    # A private helper function to transform any matrix X into
    # the polynomial + exp basis
    def __polyexpBasis(self, X):
        # poly columns
        Z = np.ones((X.shape[0], 1))
        for i in range(1, self.p + 1):
            temp_X = np.power(X, i)
            Z = np.append(Z, temp_X, axis=1)
        # exp columns
        exp_k = np.exp(X * self.k)
        Z = np.append(Z, exp_k, axis=1)
        return Z


# Least Squares with ploy basis and exp basis
class LeastSquaresPolyLog:
    def __init__(self, p, k):
        self.leastSquares = LeastSquares()
        self.p = p
        self.k = k

    def fit(self, X, y):
        n, d = X.shape
        Z = self.__polyexpBasis(X)
        self.leastSquares.fit(Z, y)

    def predict(self, X):
        Z = self.__polyexpBasis(X)
        return self.leastSquares.predict(Z)

    # A private helper function to transform any matrix X into
    # the polynomial + exp basis
    def __polyexpBasis(self, X):
        # poly columns
        Z = np.ones((X.shape[0], 1))
        for i in range(1, self.p + 1):
            temp_X = np.power(X, i)
            Z = np.append(Z, temp_X, axis=1)
        # exp columns
        exp_k = np.exp(X * self.k)
        Z = np.append(Z, exp_k, axis=1)
        return Z


# Least Squares with polynomial and sin basis
class LeastSquaresPolySinMulti:
    def __init__(self, p, k):
        self.leastSquares = LeastSquares()
        self.p = p
        self.k = k

    def fit(self,X,y):
        Z = self.__polySinBasis(X)
        self.w = solve(Z.T @ Z, Z.T @ y)

    def predict(self, X):
        Z = self.__polySinBasis(X)
        return Z @ self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polySinBasis(self, X):
        # poly part
        Z = np.ones((X.shape[0], 1))
        for i in range(1, self.p+1):
            temp_X = np.power(X, i)
            Z = np.append(Z, temp_X, axis=1)

        # sin part
        for i in range(1, self.k+1):
            temp_X = np.sin(self.k * X)
            Z = np.append(Z, temp_X, axis=1)
        return Z
