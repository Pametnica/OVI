import csv
import math
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder


def readFile(file_name):
    with open(file_name) as car_csv:
        reader = csv.reader(car_csv, delimiter=',')
        data = list(reader)[1:]
    return data


if __name__ == '__main__':
    car_data = readFile('car.csv')

    encoder = OrdinalEncoder()
    encoder.fit([car_data[i][:-1] for i in range(0, len(car_data))])

    train_data = car_data[:math.ceil(0.7 * len(car_data))]
    test_data = car_data[math.ceil(0.7 * len(car_data)):]

    X = [row[:-1] for row in train_data]
    X = encoder.transform(X)
    Y = [row[-1] for row in train_data]

    bayes = CategoricalNB()
    bayes.fit(X, Y)

    acc = 0
    test_data_transform = encoder.transform([row[:-1] for row in test_data])

    for i in range(0, len(test_data)):
        actual_class = test_data[i][-1]
        predicted = bayes.predict([test_data_transform[i]])
        if predicted[0] == actual_class:
            acc += 1
    print(acc/len(test_data))
