from os.path import isfile

from keras import Sequential, optimizers
from keras.layers import Dense
from keras.models import load_model

from dataset_process.get_data import get_train_data, get_evaluation_data
from evaluation.evaluation import evaluate_keras_neural_network


def learn_nn(model):
    data, target = get_train_data()

    model = Sequential()
    model.add(Dense(80, input_dim=68, activation='tanh'))
    model.add(Dense(70, activation='relu'))
    model.add(Dense(60, activation='tanh'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(40, activation='tanh'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(10, activation='tanh'))
    model.add(Dense(1, activation='relu'))

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

    model.fit(data, target, epochs=10, batch_size=10, verbose=True)

    data, target = get_evaluation_data()

    model.evaluate(data, target)

    return model


if isfile('keras_model.h5'):
    print("Loading model from disk")
    nn_model = load_model('keras_model.h5')
    print("Loaded model from disk")
else:
    print("Creating new neural network")
    nn_model = Sequential()
    nn_model = learn_nn(nn_model)
    nn_model.save('keras_model.h5')
    print("Neural network saved on disk")

# data, target = get_test_data()

evaluate_keras_neural_network(nn_model)

# predictions = model.predict(data)
