import os.path

import numpy as np
from nimblenet.activation_functions import ReLU_function, tanh_function
from nimblenet.cost_functions import cross_entropy_cost
from nimblenet.data_structures import Instance
from nimblenet.learning_algorithms import scaled_conjugate_gradient
from nimblenet.neuralnet import NeuralNet
from sklearn.preprocessing import StandardScaler

from dataset_process.data_io import get_train_data, get_validation_data
from evaluation.evaluation import evaluate_neural_network

network_file = 'scaled_conjugate_gradient.pkl'


def initialize_network():
    # Train
    data, target = get_train_data()

    if data.dtype != float:
        data = StandardScaler().fit_transform(data)
        print 'Dataset scaled'

    train = []
    for i in range(len(data)):
        temp = np.array(data[i], dtype='float64')
        train.append(Instance(temp, [float(target[i])]))

    # Evaluation
    data, target = get_validation_data()

    if data.dtype != float:
        data = StandardScaler().fit_transform(data, target)
        print 'Dataset scaled'

    evaluation = []
    for i in range(len(data)):
        temp = np.array(data[i], dtype='float64')
        evaluation.append(Instance(temp, [float(target[i])]))

    settings = {
        "n_inputs": len(data[0]),
        "layers": [(80, tanh_function),
                   (70, ReLU_function),
                   (60, tanh_function),
                   (50, ReLU_function),
                   (40, tanh_function),
                   (30, ReLU_function),
                   (20, tanh_function),
                   (10, tanh_function),
                   (1, ReLU_function)]
    }

    temp_network = NeuralNet(settings)
    training_set = train
    test_set = evaluation
    cost_function = cross_entropy_cost

    scaled_conjugate_gradient(
        temp_network,  # the network to train
        training_set,  # specify the training set
        test_set,  # specify the test set
        cost_function,  # specify the cost function to calculate error
        print_rate=1,
        save_trained_network=True
    )

    return temp_network


def load_network():
    return NeuralNet.load_network_from_file(network_file)


network = None
if os.path.exists(network_file):
    network = load_network()
else:
    network = initialize_network()


evaluate_neural_network(network)
