import matplotlib.pyplot as p

# TODO ALTER THIS DATA YOURS ONES
x = [
    -0.45, -0.93, -0.90, 0.58, 2.56, -0.84, -0.01, 0.55, 1.34, -0.21, 1.06, -0.27, 0.11, 0.02, 2.07, 1.04,
    0.46, 1.04, -0.22, -2.18, 1.74, 0.90, 0.83, 8.86, 8.63, -0.47, 0.99, -0.86, -1.48, 1.23, 0.68, 0.87,
    0.87, 0.27, -0.84, -0.16, 0.95, -1.73, 0.71, -1.26, 1.50, -1.57, 0.31, 0.58, -0.66, -0.18, -1.24, -0.04, 0.13, 0.52
]

y = [
    3.91, 5.31, 4.87, 2.04, -2.43, 5.12, 2.95, 1.57, 0.30, 3.29, -0.08, 4.23, 2.15,
    2.71, -2.66, 0.27, 1.83, 1.36, 3.52, 7.10, -0.63, 0.06, 0.46, 1.03, 1.04, 3.63,
    1.09, 5.15, 6.28, 0.43, 0.53, 1.15, 0.68, 1.38, 4.37, 3.81, 0.66, 7.17, 0.84, 4.58, 0.05, 7.27,
    1.67, 1.25, 5.05, 4.25, 6.41, 2.51, 2.30, 0.96
]


##############################################################
################### PEARSON ##################################
##############################################################


def avg(array):
    """
    :param array: Массив с числами
    :return: среднее значение
    """
    return sum(array) / len(array)


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


# TODO Replace this with  scipy.stats.pearsonr or yours implementation
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


print("Pearson coefficient is " + str(pearson(x, y)))
print()

##############################################################
################### SPEARMAN #################################
##############################################################

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

# TODO Replace this with scipy.stats.spearmanr or yours implementation
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


print("Spearman coefficient is " + str(spearman(x, y)))
print()

##############################################################
######### Гипотеза об отсутствии статистической связи ########
##############################################################

# Read more about this topic:
# http://www.machinelearning.ru/wiki/index.php?title=%D0%9A%D0%BE%D1%8D%D1%84%D1%84%D0%B8%D1%86%D0%B8%D0%B5%D0%BD%D1%82_%D0%BA%D0%BE%D1%80%D1%80%D0%B5%D0%BB%D1%8F%D1%86%D0%B8%D0%B8_%D0%9F%D0%B8%D1%80%D1%81%D0%BE%D0%BD%D0%B0

# T-statistics pearson coefficient
tsta = pearson(x, y) * ((50 - 2) / (1 - pearson(x, y) ** 2)) ** (1 / 2)
stydent_kvantil005 = 2.0106348
print("t-статистика для выбоки равна " + str(tsta))
if tsta < stydent_kvantil005 and -1 * stydent_kvantil005 < tsta:
    print("Гипотеза о наличии статистической"
          " связи исходя из коэфициента Пирсона принимается")
else:
    print("Гипотеза о наличии статистической"
          " связи исходя из коэфициента Пирсона не принимается")

# T-statistics spearman coefficient
tsta = spearman(x, y) * ((50 - 2) / (1 - spearman(x, y) ** 2)) ** (1 / 2)
stydent_kvantil005 = 2.0106348
print("t-статистика для выбоки равна " + str(tsta))
if tsta < stydent_kvantil005 and -1 * stydent_kvantil005 < tsta:
    print("Гипотеза о наличии статистической"
          " связи исходя из коэфициента Спирмана принимается")
else:
    print("Гипотеза о наличии статистической"
          " связи исходя из коэфициента Спирмана не принимается")

##############################################################
######### Построение графика #################################
##############################################################

# TODO thinking about results
p.scatter(x, y)
p.show()
