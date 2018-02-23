from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from dataset_process.data_io import get_train_data
from evaluation.evaluation import evaluate
from global_variables import get_transformed_dir

method = KNeighborsClassifier(n_neighbors=3)

dataset_location = get_transformed_dir()

data, target = get_train_data(dataset_location)

scaled = False
if data.dtype != float:
    data = StandardScaler().fit_transform(data)
    scaled = True
    print 'Dataset scaled'

evaluate(method.fit(data, target),
         method_name='K Nearest Neighbour',
         scaled=scaled,
         dataset_location=dataset_location)
