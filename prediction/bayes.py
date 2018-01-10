import sklearn.naive_bayes as nb

from evaluation.evaluation import evaluate
from dataset_process.get_data import get_train_data

gnb = nb.GaussianNB()

data, target = get_train_data()

evaluate(gnb.fit(data, target))

# print("Number of mislabeled points out of a total %d points : %d"
#       % (len(data), (target != y_pred).sum()))
