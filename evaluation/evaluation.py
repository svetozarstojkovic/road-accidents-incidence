from sklearn.metrics import f1_score

from dataset_process.get_data import get_test_data


def evaluate(method):

    data, target = get_test_data()

    y_pred = method.predict(data)

    print 'F_score micro: ' + str(f1_score(target, y_pred, average='micro'))
    print 'F_score macro: ' + str(f1_score(target, y_pred, average='macro'))
    print 'F_score weighted: ' + str(f1_score(target, y_pred, average='weighted'))
