import csv

from sklearn.neighbors import KNeighborsClassifier

from dataset_process.get_data import get_train_data
from evaluation.evaluation import evaluate

method = KNeighborsClassifier(n_neighbors=3)

data, target = get_train_data()

evaluate(method.fit(data, target))
