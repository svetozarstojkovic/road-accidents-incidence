from os.path import isfile

from keras import Sequential, optimizers
from keras.layers import Dense
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

from dataset_process.data_io import get_train_data, get_validation_data
from evaluation.evaluation import evaluate_sequential_nn
from global_variables import DatasetTypes

dataset_location = DatasetTypes.TRANSFORMED_DIR.value


def learn_nn():
    data, target = get_train_data(dataset_location)

    if data.dtype != float:
        print 'Scaling dataset'
        data = StandardScaler().fit_transform(data)

    model = Sequential()
    model.add(Dense(80, input_dim=data.shape[1], activation='tanh'))
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

    data_validate, target_validate = get_validation_data(dataset_location)

    if data.dtype != float:
        print 'Scaling dataset'
        data_validate = StandardScaler().fit_transform(data_validate)

    model.fit(data, target,
              epochs=10,
              batch_size=10,
              validation_data=(data_validate, target_validate),
              verbose=True)

    return model


if isfile('keras_model.h5'):
    print("Loading model from disk")
    nn_model = load_model('keras_model.h5')
    print("Loaded model from disk")
else:
    print("Creating new neural network")
    nn_model = learn_nn()
    nn_model.save('keras_model.h5')
    print("Neural network saved on disk")

evaluate_sequential_nn(nn_model, dataset_location=dataset_location)
