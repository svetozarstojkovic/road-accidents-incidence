import csv

data = []
with open('../dataset/dataset.csv', 'rb') as full_file:
    reader = csv.reader(full_file)
    for row in reader:
        data.append(row)

    header = data[0]
    data = data[1:]

    with open('../dataset/train.csv', 'w') as train_file:
        train_file.write(",".join(header))
        train_file.write('\n')
        for index in range(1, int((len(data)) * 0.8)):
            train_file.write(",".join(data[index]))
            train_file.write('\n')

    with open('../dataset/evaluate.csv', 'w') as evaluate_file:
        evaluate_file.write(",".join(header))
        evaluate_file.write('\n')
        for index in range(int((len(data)) * 0.8), int((len(data)) * 0.9)):
            evaluate_file.write(",".join(data[index]))
            evaluate_file.write('\n')

    with open('../dataset/test.csv', 'w') as test_file:
        test_file.write(",".join(header))
        test_file.write('\n')
        for index in range(int((len(data)) * 0.9), len(data)):
            test_file.write(",".join(data[index]))
            test_file.write('\n')
