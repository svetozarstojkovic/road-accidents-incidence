import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler

from dataset_process.data_io import get_data, write_to_csv
from global_variables import get_k_best_dir, get_pca_dir, get_transformed_dir


def process_pca():

    if not os.path.exists(get_pca_dir()):
        os.makedirs(get_pca_dir())

    pca = PCA(0.95)  # 0.95 means that the maximum number of columns will be removed and accuracy will stay above 95%

    data, target, headers = get_data(get_transformed_dir() + 'dataset.csv')

    data = StandardScaler().fit_transform(data)
    pca.fit(data)
    pca_data = pca.transform(data)

    plt.semilogy(pca.explained_variance_ratio_, '--o')
    plt.show()

    pca_headers = ['target']
    for i in range(pca.n_components_):
        pca_headers.append('PCA_component_'+str(i + 1))

    write_to_csv(get_pca_dir() + 'dataset.csv', pca_data, target, pca_headers)

    print pca


def select_k_best():

    if not os.path.exists(get_k_best_dir()):
        os.makedirs(get_k_best_dir())

    data, target, headers = get_data(get_transformed_dir() + 'dataset.csv')

    # data = normalize(data, axis=0, norm='max')

    data = SelectKBest(chi2, k=40).fit_transform(data, target)

    headers = ['target']
    for i in range(data.shape[1]):
        headers.append('PCA_component_' + str(i + 1))

    write_to_csv(get_k_best_dir() + 'dataset.csv', data, target, headers)


process_pca()
# select_k_best()
