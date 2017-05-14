import csv
import numpy
import numpy.linalg as linal


def read_column_from_csv(column_number, file):
    column_array = []
    # read file
    with open(file) as f:
        reader = csv.reader(f)

        column_array = [row[column_number] for row in reader]

        # pop string element like 'x2_1'
        column_array.pop(0)

        # make values float
        column_array = list(map(lambda x: float(x), column_array))

    return column_array


varinat = 10
varinat -= 1

x_2 = read_column_from_csv(column_number=0 + varinat * 4, file='data/4problem.csv')
x_3 = read_column_from_csv(column_number=1 + varinat * 4, file='data/4problem.csv')
x_4 = read_column_from_csv(column_number=2 + varinat * 4, file='data/4problem.csv')
y_1 = read_column_from_csv(column_number=3 + varinat * 4, file='data/4problem.csv')

len_of_data = len(y_1)

# vector of MSE coefficients is (Xt * X) ^-1 * Xt * Y

# 1. make vector-column
X = numpy.ones((len_of_data, 1), dtype=float)

X_2 = numpy.array([x_2]).T
X_3 = numpy.array([x_3]).T
X_4 = numpy.array([x_4]).T
Y_1 = numpy.array([y_1]).T

X = numpy.concatenate((X, X_2), axis=1)
X = numpy.concatenate((X, X_3), axis=1)
X = numpy.concatenate((X, X_4), axis=1)

print(X)

# (Xt * X)
# numpy.dot is a matrix multiplication

X_step1 = numpy.dot(X.T, X)

# (Xt * X) ^-1

X_step2 = linal.inv(X_step1)

# (Xt * X) ^-1 * Xt

X_step3 = numpy.dot(X_step2, X.T)

# (Xt * X) ^-1 * Xt * Y

coef_vector = numpy.dot(X_step3, Y_1)

print(X_step1)
print(X_step2)
print(X_step3)
print(coef_vector)
