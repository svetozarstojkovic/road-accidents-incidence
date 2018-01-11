import csv

import numpy as np

data = []
print 'Starting to load dataset'
with open('../dataset/dataset.csv', 'rb') as full_file:
    reader = csv.reader(full_file)
    for row in reader:
        data.append(row)

print 'before creating the array of arrays'
data = np.array([np.array(xi) for xi in data])
print data.shape

data = np.delete(data, 0, 1)
data = np.delete(data, 30, 1)

print data.shape

