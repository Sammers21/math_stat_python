import csv


def mk_data_var(variant):
    col_store = []

    col_store.append(read_column_from_csv(0 + (variant - 1) * 5, 'data/6problem.csv', type='r', pop=False))
    col_store.append(read_column_from_csv(1 + (variant - 1) * 5, 'data/6problem.csv', type='r', pop=False))
    col_store.append(read_column_from_csv(2 + (variant - 1) * 5, 'data/6problem.csv', type='r', pop=False))
    col_store.append(read_column_from_csv(3 + (variant - 1) * 5, 'data/6problem.csv', type='r', pop=False))
    col_store.append(read_column_from_csv(4 + (variant - 1) * 5, 'data/6problem.csv', type='r', pop=False))

    res = ''
    for i in range(len(col_store[0])):
        st = ''
        for y in range(5):
            st += col_store[y][i]

        if len(st) >= 4:
            for y in range(4):
                res += col_store[y][i] + ','

            res += col_store[4][i] + '\n'

    write_file('data/6problem_{}.csv'.format(variant), res)


def write_file(filename, text):
    with open(filename, 'w') as f:
        f.write(text)


def read_column_from_csv(column_number, file, type='f', pop=True):
    column_array = []
    # read file
    with open(file) as f:
        reader = csv.reader(f)

        column_array = [row[column_number] for row in reader]

        # pop string element like 'x2_1'
        if pop:
            column_array.pop(0)

        # make values float
        print(column_array)
        if type == 'f':
            column_array = list(map(lambda x: float(x), column_array))

    return column_array


def avg(array):
    """
    :param array: Массив с числами
    :return: среднее значение
    """
    return sum(array) / len(array)


def pearson(x, y):
    """
    Функция подсчёта коэфициета пирсона

    Аналог:
        scipy.stats.pearsonr(x, y)[0] <---->  pearson(x, y)

    Подробнее:
        http://www.machinelearning.ru/wiki/index.php?title=%D0%9A%D0%BE%D1%8D%D1%84%D1%84%D0%B8%D1%86%D0%B8%D0%B5%D0%BD%D1%82_%D0%BA%D0%BE%D1%80%D1%80%D0%B5%D0%BB%D1%8F%D1%86%D0%B8%D0%B8_%D0%9F%D0%B8%D1%80%D1%81%D0%BE%D0%BD%D0%B0

    :param x: первая случайная величина
    :param y: вторая случайная величина
    :return: коэфициент пирсона для данных велечин
    """

    cov_X_Y = covariation(x, y)
    disp_X = vdcv(x)
    disp_Y = vdcv(y)

    return cov_X_Y / (disp_X * disp_Y) ** (1 / 2)


def vdcv(arr):
    """
    Функция для подсчёта выборочной диперсии


    Аналог:
        numpy.cov(x,y)[0,0] <---> vdcv(x)
        numpy.cov(x,y)[1,1] <---> vdcv(y)


    Подробнее:
        https://ru.wikipedia.org/wiki/%D0%92%D1%8B%D0%B1%D0%BE%D1%80%D0%BE%D1%87%D0%BD%D0%B0%D1%8F_%D0%B4%D0%B8%D1%81%D0%BF%D0%B5%D1%80%D1%81%D0%B8%D1%8F

    :param arr: случайная величина(в виде массива)
    :return: выборочная дисперсия
    """
    avg_arr = avg(arr)
    sum_of_elems = 0.
    for i in range(len(arr)):
        sum_of_elems += (arr[i] - avg_arr) ** 2
    return sum_of_elems / len(arr)


def covariation(fst, sec):
    """
    Функция для подсчёта ковариации 

    Аналог:
        numpy.cov(x,y)[0,1] <-----> covariation(x, y)

    Подробнее:
        https://en.wikipedia.org/wiki/Covariance

    :param fst: первая случайная величина
    :param sec: вторая случайная величина
    :return: ковариация случайных велечин
    """
    avg_fst = avg(fst)
    avg_sec = avg(sec)

    sum = 0.
    for i in range(len(fst)):
        sum += ((fst[i] - avg_fst) * (sec[i] - avg_sec))
    return sum / len(fst)


def rank(arr, val):
    """
    Function that return rank of element in array
    :param arr: array of tuples
    :param val: element
    :return: rank of element
    """
    position = 1
    for elem in arr:
        if elem[1] == val:
            break
        position += 1
    return position


def spearman(x, y):
    """
    Function that calculate Spearman coefficient for given X and Y arrays

    Alternatives:
        scipy.stats.spearmanr(x, y)[0] <----> spearman(x,y)

    Useful link:
        https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

    :param x: X array
    :param y: Y array
    :return: Spearman coefficient for given X and Y arrays
    """
    pair_arr = []
    for i in range(len(x)):
        # append the tuple
        pair_arr.append((x[i], y[i]))

    # sort by first param
    pair_arr = sorted(pair_arr, key=lambda pair: pair[0])

    # sort by second param
    sorted_y_arr = sorted(pair_arr, key=lambda pair: pair[1])

    # replace first with its rank

    rank_Y_arr = []
    for pair in pair_arr:
        rank_Y_arr.append(rank(sorted_y_arr, pair[1]))

    sum_of_d = 0.
    for i in range(1, 51):
        sum_of_d += (i - rank_Y_arr[i - 1]) ** 2

    return 1 - 6 * sum_of_d / (50 * (50 ** 2 - 1))
