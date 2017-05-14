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


def ls(var_to_calc):
    """
    :param var_to_calc: variant from 1 to 10
    :return: array of LS(least squares) coefficients
    """

    var_to_calc -= 1

    x_2 = read_column_from_csv(column_number=0 + var_to_calc * 4, file='data/4problem.csv')
    x_3 = read_column_from_csv(column_number=1 + var_to_calc * 4, file='data/4problem.csv')
    x_4 = read_column_from_csv(column_number=2 + var_to_calc * 4, file='data/4problem.csv')
    y_1 = read_column_from_csv(column_number=3 + var_to_calc * 4, file='data/4problem.csv')

    len_of_data = len(y_1)

    # vector of MSE coefficients is (Xt * X) ^-1 * Xt * Y
    # 1. make vector-column
    x = numpy.ones((len_of_data, 1), dtype=float)

    x_2 = numpy.array([x_2]).T
    x_3 = numpy.array([x_3]).T
    x_4 = numpy.array([x_4]).T
    y_1 = numpy.array([y_1]).T

    x = numpy.concatenate((x, x_2), axis=1)
    x = numpy.concatenate((x, x_3), axis=1)
    x = numpy.concatenate((x, x_4), axis=1)

    # (Xt * X)
    # numpy.dot is a matrix multiplication
    x_step1 = numpy.dot(x.T, x)

    # (Xt * X) ^-1
    x_step2 = linal.inv(x_step1)

    # (Xt * X) ^-1 * Xt
    x_step3 = numpy.dot(x_step2, x.T)

    # (Xt * X) ^-1 * Xt * Y
    coefficient_vector = numpy.dot(x_step3, y_1)

    return coefficient_vector.T[0]


# LS(least squares)
print(ls(10))
