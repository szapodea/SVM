import math

import numpy as np
import pandas as pd

# modelIdx = represents teh model to be used
# modelIdx == 1: LR
# modelIdx == 2: SVM
def lrsvm(trainingDataFilename, testDataFilename, modelIdx):
    def lr(trainingSet, testSet):
        # some constants
        lambda_ = 0.01  # lambda value for L2 regularization
        tol = math.pow(10, -6)  # stop when l2 norm is less than this value
        iterations = 500
        eta = 0.01

        Y_Train = trainingSet['decision']
        X_Train = trainingSet.drop(columns=['decision'])

        Y_Test = testSet['decision']
        X_Test = testSet.drop(columns=['decision'])

        w = np.zeros(shape=(X_Train.shape[1]))


        for i in range(iterations):
            dot = np.dot(X_Train, w)
            #print(dot)
            y_hat = sigmoid(dot)
            gradient = np.dot(-Y_Train + y_hat, X_Train) + lambda_ * w
            weights = w - eta * gradient


            if np.linalg.norm(w - weights) < tol:
                return

            w = weights

        return w





    def sigmoid(val):
        return 1/(1 + np.exp(-val))





    trainingSet = pd.read_csv(trainingDataFilename)
    testSet = pd.read_csv(testDataFilename)
    if modelIdx == 1:
        training = lr(trainingSet=trainingSet, testSet=testSet)
        print(training)




if __name__ == '__main__':
    lrsvm('./trainingSet.csv', './testSet.csv', 1)