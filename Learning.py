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
        X_Train.insert(loc=0, column='intercept', value=intercept)
        w = np.zeros(shape=(X_Train.shape[1]))
        #print(X_Train.shape, w.shape)

        for i in range(iterations):
            new_w = w
            dot = np.dot(X_Train, w)
            y_hat = sigmoid(dot)

            gradient = np.dot(-Y_Train + y_hat, X_Train) + lambda_ * w
            new_w = new_w - eta * gradient

            l2_norm = np.linalg.norm(np.subtract(new_w, w))
            w = new_w
            if l2_norm < tol:
                break

        train_data = np.dot(X_Train, w)
        train_prediction = np.round(sigmoid(train_data))
        train_cnt = 0
        for i, j in zip(train_prediction, Y_Train):
            if i == j:
                train_cnt += 1

        test_intercept = np.ones(shape=(X_Test.shape[0], 1))
        X_Test.insert(loc=0, column='intercept', value=test_intercept)
        test_data = np.dot(X_Test, w)
        test_prediction = np.round(sigmoid(test_data))
        test_cnt = 0

        for i, j in zip(test_prediction, Y_Test):
            if i == j:
                test_cnt += 1

        print("Input: lrsvm(trainingSet.csv, testSet.csv, 1)")
        print("Training Accuracy LR: {0:.2f}".format(train_cnt / len(train_prediction)))
        print("Testing Accuracy LR: {0:.2f}".format(test_cnt / len(test_prediction)))
        return (train_cnt / len(train_prediction)), (test_cnt / len(test_prediction))

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
        N = X_Train.shape[0]

        for i in range(iterations):
            y_hat = np.dot(X_Train, w)
            dot = np.multiply(y_hat, Y_Train)
            new_w = w
            for index, row in X_Train.iterrows():
                delta_ji = 0
                if dot[index] < 1:
                    delta_ji = np.dot(Y_Train[index], row) #X_Train.loc[index,]
                delta_j = ((lambda_ * w) - delta_ji)/N
                new_w = new_w - eta * delta_j
            l2_norm = np.linalg.norm(np.subtract(new_w, w))
            w = new_w
            if l2_norm < tol:
                break

        train_data = np.dot(X_Train, w)
        train_cnt = 0
        for x, y in zip(train_data, Y_Train):
            res = -1
            if x >= 0:
                res = 1
            if res == y:
                train_cnt += 1

        intercept = np.ones(shape=(X_Test.shape[0], 1))
        X_Test.insert(loc=0, column='intercept', value=intercept)
        test_data = np.dot(X_Test, w)
        test_cnt = 0
        for x, y in zip(test_data, Y_Test):
            res = -1
            if x >= 0:
                res = 1
            if res == y:
                test_cnt += 1

        print("Input: lrsvm(trainingSet.csv, testSet.csv, 2)")
        print("Training Accuracy SVM: {0:.2f}".format(train_cnt/len(train_data)))
        print("Testing Accuracy SVM: {0:.2f}".format(test_cnt/len(test_data)))

        return train_cnt/len(train_data), test_cnt/len(test_data)








    trainingSet = pd.read_csv(trainingDataFilename)
    testSet = pd.read_csv(testDataFilename)
    if modelIdx == 1:
        training = lr(trainingSet=trainingSet, testSet=testSet)
    elif modelIdx == 2:
        training = svm(trainingSet=trainingSet, testSet=testSet)






if __name__ == '__main__':
    lrsvm('./trainingSet.csv', './testSet.csv', 1)
    lrsvm('./trainingSet.csv', './testSet.csv', 2)
