from sklearn import tree
from sklearn.preprocessing import StandardScaler

from evaluation.evaluation import evaluate
from dataset_process.data_io import get_train_data
from global_variables import DatasetTypes

method = tree.DecisionTreeClassifier()

dataset_location = DatasetTypes.TRANSFORMED_DIR.value

data, target = get_train_data(dataset_location)

scaled = False
if data.dtype != float:
    data = StandardScaler().fit_transform(data)
    scaled = True
    print 'Dataset scaled'

evaluate(method.fit(data, target),
         method_name='Decision Tree Classifier',
         scaled=scaled,
         dataset_location=dataset_location)
