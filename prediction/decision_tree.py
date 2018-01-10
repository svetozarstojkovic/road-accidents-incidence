from sklearn import tree

from evaluation.evaluation import evaluate
from dataset_process.get_data import get_train_data

method = tree.DecisionTreeClassifier()

data, target = get_train_data()

evaluate(method.fit(data, target))
