import math

import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings('ignore')


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

        # need to set intercept to 1

        intercept = np.ones(shape=(X_Train.shape[0], 1))
        X_Train = np.hstack(tup=(intercept, X_Train))
        w = np.zeros(shape=(X_Train.shape[1]))
        #print(X_Train.shape, w.shape)

        for i in range(iterations):
            dot = np.dot(w, X_Train.T)

            #print(dot.shape)

            y_hat = sigmoid(dot)
            gradient = np.dot(-Y_Train + y_hat, X_Train) + lambda_ * w
            weights = w - eta * gradient

            if np.linalg.norm(np.add(w, -weights)) < tol:
                return

            w = weights

        test_data = np.dot(X_Train, w)
        #print (test_data)
        train_prediction = np.round(sigmoid(test_data))
        train_cnt = 0
        for i, j in zip(train_prediction, Y_Train):
            # TODO: instead of checking for 1, check to see if they equal the corresponding value in X_Train
            if i == j:
                train_cnt += 1



        test_intercept = np.ones(shape=(X_Test.shape[0], 1))
        X_Test = np.hstack(tup=(test_intercept, X_Test))
        train_data = np.dot(X_Test, w)
        test_prediction = np.round(sigmoid(train_data))
        test_cnt = 0

        for i, j in zip(test_prediction, Y_Test):
            if i == j:
                test_cnt += 1

        print("Input: lrsvm(trainingSet.csv, testSet.csv, 1)")
        print("Expected Output:")
        print("Training Accuracy LR: {0:.2f}".format(train_cnt / len(train_prediction)))
        print("Testing Accuracy LR: {0:.2f}".format(test_cnt / len(test_prediction)))

    def sigmoid(val):
        return 1 / (1 + np.exp(-val))

    def svm(trainingSet, testSet):
        lambda_ = 0.01  # lambda value for L2 regularization
        tol = math.pow(10, -6)  # stop when l2 norm is less than this value
        iterations = 500
        eta = 0.5

        Y_Train = trainingSet['decision']
        Y_Train.replace(to_replace=0, value=-1, inplace=True)
        X_Train = trainingSet.drop(columns=['decision'])

        Y_Test = testSet['decision']
        Y_Test.replace(to_replace=0, value=-1, inplace=True)
        X_Test = testSet.drop(columns=['decision'])


        intercept = np.ones(shape=(X_Train.shape[0], 1))
        X_Train.insert(loc=0, column='intercept', value=intercept)
        w = np.zeros(shape=(X_Train.shape[1]))
        #print(w.shape, X_Train)

        # need to iterate up to iterations

        for i in range(iterations):
            y_hat = np.dot(X_Train, w)
            #print(Y_Train.shape, y_hat.shape)
            delta_j = np.zeros(X_Train.columns)
            if np.dot(y_hat.T, Y_Train) < 1:
                for index, col in enumerate(X_Train.columns):









    trainingSet = pd.read_csv(trainingDataFilename)
    testSet = pd.read_csv(testDataFilename)
    if modelIdx == 1:
        training = lr(trainingSet=trainingSet, testSet=testSet)
    elif modelIdx == 2:
        training = svm(trainingSet=trainingSet, testSet=testSet)









if __name__ == '__main__':
    #lrsvm('./trainingSet.csv', './testSet.csv', 1)
    lrsvm('./trainingSet.csv', './testSet.csv', 2)
