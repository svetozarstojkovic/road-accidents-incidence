import csv

import numpy as np

from global_variables import get_transformed_dir

data = []
print 'Starting to load dataset'
with open('../dataset/dataset.csv', 'rb') as full_file:
    reader = csv.reader(full_file)
    for row in reader:
        data.append(row)

print '1:6 - Data loaded from file'

header = data[0]
data = data[1:]

data = np.array([np.array(xi) for xi in data])

severity_header = header[30]
severity_data = list(data[:, 30])

print '2:6 - Created arrays'

data = np.delete(data, 55, 1)  # lsoa_of_accident_location (E01005288)
data = np.delete(data, 37, 1)  # local authority (E08000003)
data = np.delete(data, 35, 1)  # time
data = np.delete(data, 33, 1)  # date
data = np.delete(data, 30, 1)  # severity
data = np.delete(data, 28, 1)  # latitude
data = np.delete(data, 27, 1)  # longitude
data = np.delete(data, 22, 1)  # this column has all null values
data = np.delete(data, 20, 1)  # this column has all null values
data = np.delete(data, 0, 1)  # accident_index

del header[55]
del header[37]
del header[35]
del header[33]
del header[30]
del header[28]
del header[27]
del header[22]
del header[20]
del header[0]


print '3:6 - Columns deleted'

header = np.insert(header, 0, severity_header)
data = np.insert(data, 0, severity_data, axis=1)

np.place(data, data == 'NA', 'NaN')
np.place(data, data == '-1', 'NaN')

for i in range(data.shape[1]):
    temp = data[:, i]
    # temp = np.array(temp)
    non_null = temp[temp != 'NaN'].astype('float64')
    if float(len(non_null))/data.shape[0] == 0:
        print 'This index should be removed: '+str(i)
    elif float(len(non_null)) / data.shape[0] < 0.5:
        print 'This index should be altered: ' + str(i)
    else:
        temp[temp == 'NaN'] = np.median(non_null)

print '4:6 - Removed missing values and replaced with median values'

# vehicle_type = data[:, 2].reshape(-1, 1)
#
# one_hot_encoder = OneHotEncoder(sparse=False)
# one_hot_encoded = one_hot_encoder.fit_transform(vehicle_type)
# data = np.concatenate((data, one_hot_encoded), axis=1)
# data = np.delete(data, 2, 1)
# header = np.delete(header, 2)
#
# print "4:6 - Transformed categorical data"

data = data.astype(np.float)
data = data.astype(np.int64)
data = data.astype(np.str)

print '5:6 - Conversion done'
with open(get_transformed_dir() + 'dataset.csv', 'w') as trans_file:
    trans_file.write(",".join(header))
    trans_file.write('\n')
    for index in range(len(data)):
        trans_file.write(",".join(data[index]))
        trans_file.write('\n')

print '6:6 - Data written in transformed/dataset.csv file'
