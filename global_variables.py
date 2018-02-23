from os.path import dirname, abspath


def get_original_dir():
    return dirname(abspath(__file__)) + '/dataset/'


def get_pca_dir():
    return dirname(abspath(__file__)) + '/dataset/pca/'


def get_k_best_dir():
    return dirname(abspath(__file__)) + '/dataset/k_best/'


def get_transformed_dir():
    return dirname(abspath(__file__)) + '/dataset/transformed/'
