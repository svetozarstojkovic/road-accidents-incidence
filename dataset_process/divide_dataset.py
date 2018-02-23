from sklearn.model_selection import train_test_split

from dataset_process.data_io import get_data

from global_variables import DatasetTypes


def divide_dataset(root_folder):

    data, target, header = get_data(location=root_folder + 'dataset.csv')

    train_data, test_data, train_target, test_target = train_test_split(data, target,
                                                                        test_size=len(target) / 10)

    train_data, validate_data, train_target, validate_target = train_test_split(train_data, train_target,
                                                                                test_size=len(target) / 10)

    with open(root_folder + 'train.csv', 'w') as train_file:
        train_file.write(",".join(header))
        train_file.write('\n')
        for i, row in enumerate(train_data):
            train_file.write(str(train_target[i]) + "," + str(",".join(map(str, row))))
            train_file.write('\n')

    with open(root_folder + 'validate.csv', 'w') as validate_file:
        validate_file.write(",".join(header))
        validate_file.write('\n')
        for i, row in enumerate(validate_data):
            validate_file.write(str(validate_target[i]) + "," + str(",".join(map(str, row))))
            validate_file.write('\n')

    with open(root_folder + 'test.csv', 'w') as test_file:
        test_file.write(",".join(header))
        test_file.write('\n')
        for i, row in enumerate(test_data):
            test_file.write(str(test_target[i]) + "," + str(",".join(map(str, row))))
            test_file.write('\n')


divide_dataset(DatasetTypes.PCA_DIR.value)
