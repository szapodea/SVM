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

        Y_Train = trainingSet['decision']
        X_Train = trainingSet.drop(columns=['decision'])
        Y_Test = testSet['decision']
        X_Test = testSet.drop(columns=['decision'])


        w = np.zeros(shape=)


        #for i in range(iterations):





    trainingSet = pd.read_csv(trainingDataFilename)
    testSet = pd.read_csv(testDataFilename)
    if modelIdx == 1:
        lr(trainingSet=trainingSet, testSet=testSet)




if __name__ == '__main__':
    lrsvm('./trainingSet.csv', './testSet.csv', 1)