import csv
import numpy
import numpy.linalg as linal
from scipy.stats import f
from scipy.stats import t
from lib import pearson

variation = 10
k = 4
n = 40


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

    x = get_x_matrix(var_to_calc)

    # (Xt * X)
    # numpy.dot is a matrix multiplication
    x_step1 = numpy.dot(x.T, x)

    # (Xt * X) ^-1
    x_step2 = linal.inv(x_step1)

    # (Xt * X) ^-1 * Xt
    x_step3 = numpy.dot(x_step2, x.T)

    # (Xt * X) ^-1 * Xt * Y
    y_1 = read_column_from_csv(column_number=3 + (var_to_calc - 1) * 4, file='data/4problem.csv')
    y_1 = numpy.array([y_1]).T
    coefficient_vector = numpy.dot(x_step3, y_1)

    return coefficient_vector.T[0]


def get_x_matrix(var_to_calc):
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
    return x


def ess(y_arr, y_arr_explained):
    """
    :param y_arr: input array
    :param y_arr_explained: input array explained
    :return: Explained sum of squares
    """
    mean_y = sum(y_arr) / len(y_arr)

    return sum([(y_arr_explained[i] - mean_y) ** 2 for i in range(len(y_arr))])


def rss(y_arr, y_arr_explained):
    """
    :param y_arr: array
    :return: Residual sum of squares
    """

    return sum([(y_arr_explained[i] - y_arr[i]) ** 2 for i in range(len(y_arr))])


def coefficient_variance(number_of_coefficient, y_arr, y_explained, x_matrix):
    """
    :param number_of_coefficient: number from range(n)
    :param y_arr: input y
    :param y_explained: y explained array
    :param x_matrix: x matrix from docs
    :return: variance of coefficient
    """
    variance_e_2 = rss(y_arr, y_explained) / (n - k)

    # (Xt * X)^-1
    mmatrix = linal.inv(numpy.dot(x_matrix.T, x_matrix))

    v_matrix = numpy.dot(variance_e_2, mmatrix)

    return v_matrix[number_of_coefficient][number_of_coefficient]


##############################################################
######### Оцените линейную зависимость y от x2, x3 и x4#######
####### методом наименьших квадратов #########################
##############################################################

print("""1. Оцените линейную зависимость y от x2, x3 и x4
 методом наименьших квадратов""")
print()
# LS(least squares)
# b1 b2 b3 b4
coefficient_vector = ls(variation)
print("Коэфициенты b1 b2 b3 b4:")
print(coefficient_vector)
print()

##############################################################
######## Проверьте значимость регрессии в целом ##############
##############################################################

print("2. Проверьте значимость регрессии в целом")
print()
x_2 = read_column_from_csv(column_number=0 + (variation - 1) * 4, file='data/4problem.csv')
x_3 = read_column_from_csv(column_number=1 + (variation - 1) * 4, file='data/4problem.csv')
x_4 = read_column_from_csv(column_number=2 + (variation - 1) * 4, file='data/4problem.csv')
y_1 = read_column_from_csv(column_number=3 + (variation - 1) * 4, file='data/4problem.csv')

y_estimation = [coefficient_vector[0]
                + coefficient_vector[1] * x_2[i]
                + coefficient_vector[2] * x_3[i]
                + coefficient_vector[3] * x_4[i]
                for i in range(40)]

ess_ur = ess(y_1, y_estimation)
rss_ur = rss(y_1, y_estimation)

# Fisher dist
f_crit = f.ppf(0.95, k - 1, n - k)
f_real = ess_ur / (k - 1) / (rss_ur / (n - k))

print('F (95%, k-1, n-4) is {}'.format(f_crit))
print('ess / (k - 1) / (rss / (n - k)) is {}'.format(f_real))

if f_crit < f_real:
    print('Отвергаем гипотзу о значимости модели регрессии в целом (b1=b2=b3=b4=0)')
else:
    print('Принмаем гипотзу о значимости модели регрессии в целом (b1=b2=b3=b4=0)')

print()

##############################################################
######## Проверьте значимость  коэффициентов при объясняющих##
####### переменных по отдельности ############################
##############################################################

print('3. Проверьте значимость  коэффициентов при объясняющих переменных по отдельности')
print()

t_critical = t.ppf(0.95, n - k)
print('Критическое значение t~(n-k):{}'.format(t_critical))
print()

for i in range(4):
    t_val = coefficient_vector[i] / (coefficient_variance(i, y_1, y_estimation, get_x_matrix(variation))) ** (1 / 2)
    print("Гипотеза H0: коэфциент b{}=0".format(i + 1))
    print('Критерий значимости для коэфициента b{}:\t{}'.format(i + 1, t_val))
    if t_critical >= t_val >= -t_critical:
        print('\tПринмаем гипотзу о том что b{}=0'.format(i + 1))
    else:
        print('\tОтвергаем гипотзу о том что b{}=0'.format(i + 1))
    print()

##############################################################
######## Проверьте гипотезу о совместной значимости ##########
####### коэффициентов при переменных x3 и x4 #################
##############################################################

print('4. Проверьте гипотезу о совместной значимости коэффициентов при переменных x3 и x4')
print()

# it is the same model but assuming that x3 and x4 is equals to 0
# that is what we called r(Restricted by some condition like x3 = x4 = 0)

y_estimation_r = [coefficient_vector[0]
                  + coefficient_vector[1] * x_2[i]
                  for i in range(40)]

q = 2  # count of "=" in condition

# so condition is :
#  x3=x4
#  x3=0

# RSS Restricted
rss_r = rss(y_1, y_estimation_r)

f_r_critical = f.ppf(0.95, q, n - k)
f_r_real = (rss_r - rss_ur) / q / (rss_ur / (n - k))

print('F (95%, q, n-k) is {}'.format(f_crit))
print('(rss_r - rss_ur) / q / (rss_ur / (n - k)) is {}'.format(f_real))

if f_r_critical < f_r_real:
    print('Отвергаем гипотзу  о совместной значимости коэффициентов при переменных x3 и x4')
else:
    print('Принмаем гипотзу  о совместной значимости коэффициентов при переменных x3 и x4')

print()

##############################################################
######## Рассчитайте корреляционную матрицу ##################
######## для объясняющих переменных ##########################
##############################################################

print('5. Рассчитайте корреляционную матрицу для объясняющих переменных')
print()

corr_matrix = [y_1, x_2, x_3, x_4]
names = ["Y", "X2", "X3", "X4"]
tab = "\t"
print(tab + names[0] + tab + tab + names[1] + tab + tab + names[2] + tab + tab + names[3])
for i in range(4):
    print(names[i], end=tab)
    for j in range(4):
        print("%.4f" % pearson(corr_matrix[i], corr_matrix[j]), end=tab)
    print()
