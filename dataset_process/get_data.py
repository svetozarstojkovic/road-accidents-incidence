import csv
from time import time

import numpy as np


def get_whole_dataset():
    return get_data('../dataset/dataset.csv')


def get_train_data():
    return get_data('../dataset/train.csv')


def get_evaluation_data():
    return get_data('../dataset/evaluate.csv')


def get_test_data():
    return get_data('../dataset/test.csv')


def get_data(location):
    t1 = time()
    data = []
    target = []
    with open(location, 'rb') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i != 0:
                data.append(map(int, row))
    print('loaded data from csv file')
    # data = data[1:]

    # data = np.array([np.array(xi) for xi in data]).astype(np.int)

    # data = data.astype(np.int)

    # target = list(zip(*data)[0])
    # del data[:][1]
    # data = np.delete(data, 0, 1)
    # data = data[1:]

    # values = []
    # for ind in range(len(data)):
    #     print len(data[ind])
    #     if data[ind] not in values:
    #         values.append(data[ind])
    #
    # print values

    for i in range(len(data)):
        target.append(data[i][0])
        # del data[i][30]
        del data[i][0]

    print('removed columns from data')

    # for i in range(len(data)):
    #     for j in range(len(data[i])):
    #         try:
    #             data[i][j] = int(data[i][j])
    #         except ValueError:
    #             print data[i][j]
    #             data[i][j] = 0
    #
    #         # if data[i][j] < 0:
    #         #     data[i][j] = 0

    # data = [[int(float(j)) for j in i] for i in data]

    t2 = time()

    print 'This method took: '+str(t2 - t1)+' sec'

    return data, target
