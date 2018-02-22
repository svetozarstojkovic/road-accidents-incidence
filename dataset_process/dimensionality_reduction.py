import csv

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


def process_pca():
    pca = PCA(n_components=79)
    data = []
    with open('../dataset/dataset_transformed.csv', 'rb') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i != 0:
                data.append(map(int, row))
    print 'Dataset read'
    pca.fit(data)

    plt.semilogy(pca.explained_variance_ratio_, '--o')
    plt.show()

    print pca


process_pca()
