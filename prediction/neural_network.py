import numpy as np
from nimblenet.activation_functions import sigmoid_function, ReLU_function
from nimblenet.cost_functions import cross_entropy_cost, softmax_neg_loss
from nimblenet.learning_algorithms import RMSprop,scaled_conjugate_gradient
from nimblenet.data_structures import Instance
from nimblenet.neuralnet import NeuralNet

from dataset_process.get_data import get_train_data, get_test_data

# dataset = [Instance( [0,0], [0] ), Instance( [1,0], [1] ), Instance( [0,1], [1] ), Instance( [1,1], [0] )]

#Train
data, target = get_train_data()
train = []
for i in range(len(data)):
    temp = np.array(data[i], dtype='float64')
    train.append(Instance(temp, [float(target[i])]))

#Test
data, target = get_test_data()
test = []
for i in range(len(data)):
    temp = np.array(data[i], dtype='float64')
    test.append(Instance(temp, [float(target[i])]))

settings = {
    "n_inputs": len(data[0]),
    "layers": [(2, sigmoid_function), (1, sigmoid_function)]
}

network= NeuralNet(settings)
training_set = train
test_set = test
cost_function = cross_entropy_cost


scaled_conjugate_gradient(
    network,           # the network to train
    training_set,      # specify the training set
    test_set,          # specify the test set
    cost_function,     # specify the cost function to calculate error
)