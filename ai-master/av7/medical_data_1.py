import csv
import math
from sklearn.naive_bayes import GaussianNB


def read_file(file_name):
    with open(file_name) as medical_csv:
        reader = csv.reader(medical_csv, delimiter=',')
        d = list(reader)[1:]

    return d


if __name__ == '__main__':
    data = read_file('medical_data.csv')
    data = [[int(data[i][j]) for j in range(0, len(data[i]))] for i in range(0, len(data))]

    train_data = [data[i] for i in range(0, math.ceil(0.8*len(data)))]
    test_data = [data[i] for i in range(math.ceil(0.8*len(data)), len(data))]

    X = [row[:-1] for row in train_data]
    Y = [row[-1] for row in train_data]

    bayes = GaussianNB()
    bayes.fit(X, Y)

    acc = 0

    for i in range(0, len(test_data)):
        actual_class = test_data[i][-1]
        predicted_class = bayes.predict([test_data[i][:-1]])[0]
        if predicted_class == actual_class:
            acc += 1

    print(acc / len(test_data))
