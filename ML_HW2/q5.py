############################
#                          #
# authors:                 #
# Zixi Huang(zh2313)       # 
# Neil Kumar(nk2739)       # 
# Yichen Pan(yp2450)       #
#                          #
############################

import scipy.io as sc
import numpy as np
import matplotlib.pyplot as plt
import copy

# activation function
def sigma(x):
    y = 1 / (1 + np.exp(-x))
    return y

if __name__ == '__main__':
    # load data
    loaded_file = sc.loadmat('hw2data.mat')
    X = loaded_file['X']
    Y = loaded_file['Y']

    # initialize
    number_of_data = X.shape[0]
    number_of_input = X.shape[1]
    number_of_output = Y.shape[1]
    number_of_hidden = 5
    learning_rate = 10

    # weight and bias
    W1 = np.random.random((number_of_input, number_of_hidden))
    b1 = np.random.random(number_of_hidden)
    W2 = np.random.random((number_of_hidden, number_of_output))
    b2 = np.random.random(number_of_output)

    # predict value
    Y_hat = np.zeros([number_of_data, number_of_output])
    old_Y_hat = np.zeros([number_of_data, number_of_output])

    is_converged = False

    while not is_converged:
        for data in range(number_of_data):
            Y_hat[data] = sigma(W2.T.dot(sigma(W1.T.dot(X[data]) + b1)) + b2)

        # initialize
        dW2 = np.zeros_like(W2)
        db2 = np.zeros_like(b2)
        dW1 = np.zeros_like(W1)
        db1 = np.zeros_like(b1)

        # calculate gradient
        for data in range(number_of_data):
            dW2[:, 0] += 1 / (2 * number_of_data) \
                         * 2 * (Y_hat[data] - Y[data]) \
                         * Y_hat[data] * (1 - Y_hat[data]) \
                         * sigma(W1.T.dot(X[data]) + b1)
            db2 += 1 / (2 * number_of_data) \
                   * 2 * (Y_hat[data] - Y[data]) \
                   * Y_hat[data] * (1 - Y_hat[data])

            for column in range(number_of_hidden):
                Y_hat_temp = sigma(W2[column, 0] * (sigma(W1[0, column] * (X[data]) + b1[column])) + b2)
                dW1[:, column] += 1 / (2 * number_of_data) \
                                  * 2 * (Y_hat_temp - Y[data]) \
                                  * Y_hat_temp * (1 - Y_hat_temp) \
                                  * W2[column, 0] \
                                  * sigma(W1[0, column]*(X[data]) + b1[column]) * (1 - sigma(W1[0, column]*(X[data]) + b1[column])) \
                                  * X[data]
                db1[column] += 1 / (2 * number_of_data) \
                               * 2 * (Y_hat_temp - Y[data]) \
                               * Y_hat_temp * (1 - Y_hat_temp) \
                               * W2[column, 0] \
                               * sigma(W1[0, column]*(X[data]) + b1[column]) * (1 - sigma(W1[0, column]*(X[data]) + b1[column]))

        # update weight
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1

        # display learning process
        print(sum((Y_hat - Y)**2), sum((old_Y_hat - Y)**2))

        # converge criterion
        if abs(sum((Y_hat - Y)**2) - sum((old_Y_hat - Y)**2)) / sum((old_Y_hat - Y)**2) < 0.001:
            is_converged = True

        # if learning rate is too high
        if sum((Y_hat - Y)**2) - sum((old_Y_hat - Y)**2) > 0:
            learning_rate /= 2

        # store old value to check convergence
        old_Y_hat = copy.copy(Y_hat)

    # plot
    plt.figure()
    plt.scatter(X, Y, label='Original Distribution')
    plt.scatter(X, Y_hat, label='Predicted Distribution')
    plt.legend()
    plt.show()
