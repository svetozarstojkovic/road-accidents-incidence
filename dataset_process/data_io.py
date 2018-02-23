import csv
from time import time

import numpy as np

from global_variables import DatasetTypes


def get_original_dataset():
    return get_data('../dataset/dataset.csv')


def get_train_data(root_dir=DatasetTypes.TRANSFORMED_DIR.value):
    data, target, header = get_data(root_dir + 'train.csv')
    return data, target


def get_validation_data(root_dir=DatasetTypes.TRANSFORMED_DIR.value):
    data, target, header = get_data(root_dir + 'validate.csv')
    return data, target


def get_test_data(root_dir=DatasetTypes.TRANSFORMED_DIR.value):
    data, target, header = get_data(root_dir + 'test.csv')
    return data, target


# returns data, target, header
def get_data(location):
    t1 = time()
    data = []
    target = []
    header = []
    with open(location, 'rb') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                header = row
            elif i != 0:
                try:
                    data.append(map(int, row))
                except ValueError:
                    data.append(map(float, row))

    for i in range(len(data)):
        target.append(data[i][0])
        # del data[i][30]
        del data[i][0]

    data = np.array(data)

    t2 = time()

    print 'Getting data from ' + location + '\ntook: '+str(t2 - t1)+' sec\n'

    return data, target, header


def write_to_csv(location, data, target, headers):
    with open(location, 'w') as pca_file:
        pca_file.write(",".join(headers))
        pca_file.write('\n')
        for i, row in enumerate(data):
            pca_file.write(str(target[i]) + "," + str(",".join(map(str, row))))
            pca_file.write('\n')
