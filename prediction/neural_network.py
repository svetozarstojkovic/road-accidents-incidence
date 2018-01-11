import os.path

import numpy as np
from nimblenet.activation_functions import ReLU_function, tanh_function
from nimblenet.cost_functions import cross_entropy_cost
from nimblenet.data_structures import Instance
from nimblenet.learning_algorithms import scaled_conjugate_gradient
from nimblenet.neuralnet import NeuralNet

from dataset_process.get_data import get_train_data, get_evaluation_data
from evaluation.evaluation import evaluate_neural_network


def initialize_network():
    # dataset = [Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [0] )]

    # Train
    data, target = get_train_data()
    train = []
    for i in range(len(data)):
        temp = np.array(data[i], dtype='float64')
        train.append(Instance(temp, [float(target[i])]))

    # Evaluation
    data, target = get_evaluation_data()
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
    return NeuralNet.load_network_from_file('network0.pkl')


network = None
if os.path.exists('network0.pkl'):
    network = load_network()
else:
    network = initialize_network()


evaluate_neural_network(network)
