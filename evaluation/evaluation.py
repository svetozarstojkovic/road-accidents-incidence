import numpy as np
from nimblenet.data_structures import Instance
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from dataset_process.data_io import get_test_data
from global_variables import DatasetTypes


def evaluate(method, method_name, scaled=False, dataset_location=DatasetTypes.TRANSFORMED_DIR.value):

    data, target = get_test_data(dataset_location)

    if scaled:
        data = StandardScaler().fit_transform(data)
        print 'Dataset scaled'

    prediction = method.predict(data).astype(float)
    target = np.array(target, dtype=float)

    print(method_name + ':')
    print_f_measure(target, prediction)


def evaluate_neural_network(network):

    data, target = get_test_data()

    if data.dtype != float:
        data = StandardScaler().fit_transform(data)
        print 'Dataset scaled'

    test = []
    for i in range(len(data)):
        temp = np.array(data[i], dtype='float64')
        test.append(Instance(temp))

    prediction = list(np.rint(network.predict(test)).astype(int).astype(str).flatten('F'))
    target = map(np.str, target)
    print('Scaled conjugate network: ')
    print_f_measure(target, prediction)


def evaluate_sequential_nn(method, dataset_location=DatasetTypes.TRANSFORMED_DIR.value):

    data, target = get_test_data(dataset_location)

    if data.dtype != float:
        data = StandardScaler().fit_transform(data)
        print 'Dataset scaled'

    prediction = list(np.rint(method.predict(data)).astype(int).astype(str).flatten('F'))
    target = map(np.str, target)

    print('Sequential neural network: ')
    print_f_measure(target, prediction)


def print_f_measure(target, prediction):
    print('f_score micro: ' + str(f1_score(target, prediction, average='micro')))
    print('f_score macro: ' + str(f1_score(target, prediction, average='macro')))
    print('f_score weighted: ' + str(f1_score(target, prediction, average='weighted')))
