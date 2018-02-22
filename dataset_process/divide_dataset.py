import csv

import numpy as np
from sklearn.model_selection import train_test_split
from dataset_process.get_data import get_data


def divide_dataset():

    data, target, header = get_data(location='../dataset/dataset_transformed.csv')

    train_data, test_data, train_target, test_target = train_test_split(data, target,
                                                                        test_size=len(target) / 10)

    train_data, validate_data, train_target, validate_target = train_test_split(train_data, train_target,
                                                                                test_size=len(target) / 10)

    with open('../dataset/train.csv', 'w') as train_file:
        train_file.write(",".join(header))
        train_file.write('\n')
        for i, row in enumerate(train_data):
            train_file.write(str(train_target[i]) + "," + str(",".join(map(str, row))))
            train_file.write('\n')

    with open('../dataset/validate.csv', 'w') as validate_file:
        validate_file.write(",".join(header))
        validate_file.write('\n')
        for i, row in enumerate(validate_data):
            validate_file.write(str(validate_target[i]) + "," + str(",".join(map(str, row))))
            validate_file.write('\n')

    with open('../dataset/test.csv', 'w') as test_file:
        test_file.write(",".join(header))
        test_file.write('\n')
        for i, row in enumerate(test_data):
            test_file.write(str(test_target[i]) + "," + str(",".join(map(str, row))))
            test_file.write('\n')

    # data = []
    # with open('../dataset/dataset'+str(trans)+'.csv', 'rb') as full_file:
    #     reader = csv.reader(full_file)
    #     for row in reader:
    #         data.append(row)
    #
    #     header = data[0]
    #     data = data[1:]

        # with open('../dataset/train.csv', 'w') as train_file:
        #     train_file.write(",".join(header))
        #     train_file.write('\n')
        #     for index in range(1, int((len(data)) * 0.8)):
        #         train_file.write(",".join(data[index]))
        #         train_file.write('\n')
        #
        # with open('../dataset/evaluate.csv', 'w') as validate_file:
        #     validate_file.write(",".join(header))
        #     validate_file.write('\n')
        #     for index in range(int((len(data)) * 0.8), int((len(data)) * 0.9)):
        #         validate_file.write(",".join(data[index]))
        #         validate_file.write('\n')
        #
        # with open('../dataset/test.csv', 'w') as test_file:
        #     test_file.write(",".join(header))
        #     test_file.write('\n')
        #     for index in range(int((len(data)) * 0.9), len(data)):
        #         test_file.write(",".join(data[index]))
        #         test_file.write('\n')


divide_dataset()
# divide_dataset('')
