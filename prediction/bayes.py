import sklearn.naive_bayes as nb

from evaluation.evaluation import evaluate
from dataset_process.data_io import get_train_data
from global_variables import get_transformed_dir

gnb = nb.GaussianNB()

dataset_location = get_transformed_dir()

data, target = get_train_data(dataset_location)

evaluate(gnb.fit(data, target),
         method_name='Naive Bayes',
         scaled=False,
         dataset_location=dataset_location)
