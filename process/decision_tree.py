import csv

from sklearn import tree
from sklearn.metrics import f1_score

data = []
target = []
method = tree.DecisionTreeClassifier()

with open('../dataset/dataset.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        data.append(row)
print 'loaded data from csv file'
data = data[1:]

for i in range(len(data)):
    target.append(data[i][30])
    del data[i][30]
    del data[i][0]

print 'removed columns from data'

for i in range(len(data)):
    for j in range(len(data[i])):
        try:
            data[i][j] = int(data[i][j])
        except ValueError:
            data[i][j] = 0

        if data[i][j] < 0:
            data[i][j] = 0

print 'converted data to int, and removed negative data'

y_pred = method.fit(data, target).predict(data)

print 'F_score micro: '+str(f1_score(target, y_pred, average='micro'))
print 'F_score macro: '+str(f1_score(target, y_pred, average='macro'))
print 'F_score weighted: '+str(f1_score(target, y_pred, average='weighted'))

# print("Number of mislabeled points out of a total %d points : %d"
#       % (len(data), (target != y_pred).sum()))
