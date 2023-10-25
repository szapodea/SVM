import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

import NBC as nbc

import warnings
warnings.filterwarnings('ignore')


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
    # print(w.shape, X_Train)

    # need to iterate up to iterations
    N = X_Train.shape[0]

    for i in range(iterations):
        y_hat = np.dot(X_Train, w)
        dot = np.multiply(y_hat, Y_Train)
        new_w = w
        for index, row in X_Train.iterrows():
            delta_ji = 0
            if dot[index] < 1:
                delta_ji = np.dot(Y_Train[index], row)  # X_Train.loc[index,]
            delta_j = ((lambda_ * w) - delta_ji) / N
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

    Y_Test.replace(to_replace=-1, value=0, inplace=True)
    Y_Train.replace(to_replace=-1, value=0, inplace=True)

    return train_cnt / len(train_data), test_cnt / len(test_data)
def cross_validation(trainingFile, trainingFileNBC):
    data = pd.read_csv(trainingFile)
    data = data.sample(random_state=18, frac=1)

    # divide data in 10 samples
    sample_length = len(data)/10
    samples = {}
    for i in range(10):
        samples['S{0}'.format(i + 1)] = data[int(sample_length * i):int(sample_length * (i + 1))]


    t_frac = [.025, .05, .075, .1, .15, .2]
    mean_train_lr = []
    std_train_lr = []
    err_train_lr = []
    mean_test_lr = []
    std_test_lr = []
    err_test_lr = []
    mean_train_svm = []
    std_train_svm = []
    err_train_svm = []
    mean_test_svm = []
    std_test_svm = []
    err_test_svm = []
    # Handling SVM and LR
    for frac in t_frac:
        lr_train = []
        lr_test = []
        svm_train = []
        svm_test = []
        i = 0
        for sample in samples.keys():
            print('.')
            test_set = samples[sample]
            S_c = data.drop(data.index[range(int(sample_length * i), int(sample_length * (i + 1)))], axis=0)
            i += 1
            train_set = S_c.sample(frac=frac, random_state=32)
            lr1, lr2 = lr(trainingSet=train_set, testSet=test_set)
            lr_train.append(lr1)
            lr_test.append(lr2)
            svm1, svm2 = svm(trainingSet=train_set, testSet=test_set)
            svm_train.append(svm1)
            svm_test.append(svm2)

        mean_train_lr.append(np.mean(lr_train))
        mean_test_lr.append(np.mean(lr_test))
        std_train_lr.append(np.std(lr_train))
        std_test_lr.append(np.std(lr_test))
        mean_train_svm.append(np.mean(svm_train))
        mean_test_svm.append(np.mean(svm_test))
        std_train_svm.append(np.std(svm_train))
        std_test_svm.append(np.std(svm_test))

    for i in range(len(mean_test_svm)):
        err_train_lr.append(std_train_lr[i] / np.sqrt(10))
        err_test_lr.append(std_test_lr[i] / np.sqrt(10))
        err_train_svm.append(std_train_svm[i] / np.sqrt(10))
        err_test_svm.append(std_test_svm[i] / np.sqrt(10))


    nbcData = pd.read_csv(trainingFileNBC)
    nbcData = nbcData.sample(random_state=18, frac=1)
    nbcsamples = {}
    for i in range(10):
        # print(int(sample_length * i + 1), int(sample_length * (i + 1)))
        nbcsamples['S{0}'.format(i + 1)] = nbcData[int(sample_length * i):int(sample_length * (i + 1))]
    # NBC Part
    mean_train_nbc = []
    std_train_nbc = []
    err_train_nbc = []
    mean_test_nbc = []
    std_test_nbc = []
    err_test_nbc = []
    nbc_classifier = nbc.NBC()
    for frac in t_frac:
        nbc_train = []
        nbc_test = []
        i = 0
        for nbcsample in nbcsamples.keys():
            test_set_nbc = nbcsamples[nbcsample]
            S_c = nbcData.drop(nbcData.index[range(int(sample_length * i), int(sample_length * (i + 1)))], axis=0)
            i += 1
            train_set_nbc = S_c.sample(frac=frac, random_state=32)

            nbc1, nbc2 = nbc_classifier.nbc(train_set_nbc, test_set_nbc)
            nbc_train.append(nbc1)
            nbc_test.append(nbc2)

        mean_train_nbc.append(np.mean(nbc_train))
        mean_test_nbc.append(np.mean(nbc_test))
        std_train_nbc.append(np.std(nbc_train))
        std_test_nbc.append(np.std(nbc_test))

    for i in range(len(mean_train_nbc)):
        err_train_nbc.append(std_train_nbc[i] / np.sqrt(10))
        err_test_nbc.append(std_test_nbc[i] / np.sqrt(10))

    size_lst = []
    for frac in t_frac:
        size_lst.append((5200 * .9) * frac)



    plt.errorbar(x=size_lst, y=mean_test_lr, yerr=err_test_lr, ecolor='grey', color='red', label="LR Averages")
    plt.errorbar(x=size_lst, y=mean_test_svm, yerr=err_test_svm, ecolor='grey', color='blue', label="SVM Averages")
    plt.errorbar(x=size_lst, y=mean_test_nbc, yerr=err_test_nbc, ecolor='grey', color='green', label="NBC Averages")
    plt.ylabel("Model Accuracy")
    plt.xlabel("Training Data Size")
    plt.title("Average Test Accuracy vs Training Data Size")
    plt.legend()
    plt.show()




def test(data1, data2):
    mean_test_diff = []
    mean_5_percent =[]
    for i in range(len(data1)):
        mean_test_diff.append(data1[i] - data2[i])
        mean_5_percent.append(.05)

    t_test, p_value = sp.ttest_ind(mean_test_diff, mean_5_percent, equal_var=False)
    print('T-Statistic: {0:.4f}'.format(t_test / 2))
    print("P-Value: {0:.4f}".format(p_value))
    if p_value < .05:
        print("Because the p-value is less than the significance level of .05, we reject the null hypothesis")
    else:
        print("Because the p-value is not less than the significance level of .05, we fail to reject the null hypothesis")





















if __name__ == '__main__':
    #cross_validation(trainingFile='./trainingSet.csv', trainingFileNBC='./trainingSet-binned.csv')
    mean_test_lr =[0.6511538461538462, 0.6519230769230769, 0.6419230769230769, 0.6359615384615385, 0.6475000000000001, 0.6503846153846153]
    mean_test_svm = [0.5723076923076923, 0.5517307692307691, 0.5684615384615386, 0.5603846153846155, 0.5653846153846154, 0.5663461538461538]
    test(mean_test_lr, mean_test_svm)