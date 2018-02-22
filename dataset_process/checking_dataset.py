from dataset_process.get_data import get_data


def view_severity_distribution(location='../dataset/dataset_transformed.csv'):
    data, target, header = get_data(location)
    values = {}
    for value in target:
        if value in values:
            values[value] += 1
        else:
            values[value] = 1

    for key, value in values.iteritems():
        print str(key) + ': ' + str(round(float(values[key]) / len(target) * 100, 2)) + '%'


view_severity_distribution()

view_severity_distribution(location='../dataset/train.csv')
view_severity_distribution(location='../dataset/test.csv')
view_severity_distribution(location='../dataset/validate.csv')
