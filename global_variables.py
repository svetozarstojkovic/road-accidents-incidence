import os
from enum import Enum
from os.path import dirname, abspath


class DatasetTypes(Enum):
    ORIGINAL_DIR = dirname(abspath(__file__)) + '/dataset/'
    PCA_DIR = dirname(abspath(__file__)) + '/dataset/pca/'
    K_BEST_DIR = dirname(abspath(__file__)) + '/dataset/k_best/'
    TRANSFORMED_DIR = dirname(abspath(__file__)) + '/dataset/transformed/'
