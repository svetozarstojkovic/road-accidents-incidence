from dataset_process.data_io import get_data


# this script checks distribution of accident severity in train, validate and test
from global_variables import DatasetTypes


def view_severity_distribution(location='../dataset/transformed/dataset.csv'):
    data, target, header = get_data(location)
    values = {}
    for value in target:
        if value in values:
            values[value] += 1
        else:
            values[value] = 1

    for key, value in values.iteritems():
        print str(key) + ': ' + str(round(float(values[key]) / len(target) * 100, 2)) + '%'


def print_distribution_for_train_validate_test(root_dir):
    view_severity_distribution(location=root_dir + 'dataset.csv')

    view_severity_distribution(location=root_dir + 'train.csv')
    view_severity_distribution(location=root_dir + 'validate.csv')
    view_severity_distribution(location=root_dir + 'test.csv')


print_distribution_for_train_validate_test(DatasetTypes.TRANSFORMED_DIR.value)
