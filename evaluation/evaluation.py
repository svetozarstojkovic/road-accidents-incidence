import numpy as np
from nimblenet.data_structures import Instance
from sklearn.metrics import f1_score

from dataset_process.get_data import get_test_data


def evaluate(method):

    data, target = get_test_data()

    prediction = method.predict(data).astype(float)
    target = np.array(target, dtype=float)

    print('F_score micro: ' + str(f1_score(target, prediction, average='micro')))
    print('F_score macro: ' + str(f1_score(target, prediction, average='macro')))
    print('F_score weighted: ' + str(f1_score(target, prediction, average='weighted')))


def evaluate_neural_network(network):
    # Test
    data, target = get_test_data()
    test = []
    for i in range(len(data)):
        temp = np.array(data[i], dtype='float64')
        test.append(Instance(temp))

    prediction = list(np.rint(network.predict(test)).astype(int).astype(str).flatten('F'))

    print('F_score micro: ' + str(f1_score(target, prediction, average='micro')))
    print('F_score macro: ' + str(f1_score(target, prediction, average='macro')))
    print('F_score weighted: ' + str(f1_score(target, prediction, average='weighted')))


def evaluate_keras_neural_network(method):

    data, target = get_test_data()

    prediction = list(np.rint(method.predict(data)).astype(int).astype(str).flatten('F'))

    print('F_score micro: ' + str(f1_score(target, prediction, average='micro')))
    print('F_score macro: ' + str(f1_score(target, prediction, average='macro')))
    print('F_score weighted: ' + str(f1_score(target, prediction, average='weighted')))
