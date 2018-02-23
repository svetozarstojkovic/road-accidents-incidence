import sklearn.naive_bayes as nb

from evaluation.evaluation import evaluate
from dataset_process.data_io import get_train_data
from global_variables import DatasetTypes

gnb = nb.GaussianNB()

dataset_location = DatasetTypes.TRANSFORMED_DIR.value

data, target = get_train_data(dataset_location)

evaluate(gnb.fit(data, target),
         method_name='Naive Bayes',
         scaled=False,
         dataset_location=dataset_location)
