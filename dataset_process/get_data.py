import csv


def get_whole_dataset():
    return get_data('../dataset/dataset.csv')


def get_train_data():
    return get_data('../dataset/train.csv')


def get_evaluation_data():
    return get_data('../dataset/evaluate.csv')


def get_test_data():
    return get_data('../dataset/test.csv')


def get_data(location):
    data = []
    target = []
    with open(location, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    print 'loaded data from csv file'
    data = data[1:]

    for i in range(len(data)):
        target.append(data[i][30])
        del data[i][30]
        del data[i][0]

    print 'removed columns from data'

    for i in range(len(data)):
        for j in range(len(data[i])):
            try:
                data[i][j] = int(data[i][j])
            except ValueError:
                data[i][j] = 0

            if data[i][j] < 0:
                data[i][j] = 0

    print 'converted data to int, and removed negative data'

    return data, target
