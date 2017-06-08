from math import sqrt
from random import randrange

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from lib import read_column_from_csv

# TODO alter this to your variant
v_number = 1

""" 
    В таблице с данными содержатся координаты x и y точек некоторого изображения
    (давайте считать, что это обработанное фото трёх монеток с зашумлением). 
    Примените к данным вашего варианта метод k-средних, перебирая число кластеров k = 2, 3, 4. 
    Для каждого k изобразите полученное разбиение на графике, выделяя кластеры разными цветами.
    По графикам оцените, насколько удалось распознать образы монеток.
    """


def draw_plot(x, y):
    """
    method draws and saves starting plot. without clustering. 
    :param x: x data set
    :param y: y data set
    :param color: 
    :return: void
    """
    plt.plot(x, y, "k.", markersize=4)
    plt.title("Starting image")
    savefig('starting_image.png')


def draw_clusters(clusters):
    """
    this methods draws all clusters and saves picture
    :param clusters: 
    :return: void
    """
    clusrs = restructure_clusters(clusters)
    colours = ['r.', 'k.', 'b.', 'g.', 'y.', 'c.']
    plt.title("Clusters")
    for i in range(len(clusrs)):
        plt.plot(clusrs[i][0], clusrs[i][1], colours[i])

    savefig('clusters_' + str(len(clusters)) + '.png')


def restructure_clusters(clusters):
    """
    since clusters looks like [[(x,y)..(x,y)],..[(x,y)..(x,y)]] we need to restructure them to use in plot
    :param clusters: 
    :return: clusters which looks like [[[x1..xn]]]
    """
    clusrs = [[] for i in range(len(clusters))]
    for i in range(len(clusters)):
        X = []
        Y = []
        for j in range(len(clusters[i])):
            X.append(clusters[i][j][0])
            Y.append(clusters[i][j][1])
        clusrs[i].append(X)
        clusrs[i].append(Y)
    return clusrs


def calc_euclid(x1, y1, x2, y2):
    """
    calcs euclid distance
    :param x1: x coordinate of point
    :param y1: y coordinate of point
    :param x2: x coordinate of center
    :param y2: y coordinate of center
    :return: euclid distance
    """
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_start_centers(k, x, y):
    """
    method helps to get random points, which will be starting centers of our clusters
    :param k: number of clusters
    :param x: x data set
    :param y: y data set
    :return: list which contains k random points
    """
    chosen = set()
    centers = []
    for i in range(k):
        _rand = randrange(len(x))
        while _rand in chosen:
            _rand = randrange(len(x))
        chosen.add(_rand)
        centers.append([x[_rand], y[_rand]])
    return centers


def clustering(k, x, y):
    """
    clustering method
    :param k: number of clusters
    :param x: x data set 
    :param y: y data set
    :return: k clusters
    """
    # clusters - list of k lists, where k - number of clusters, given by task
    clusters = [[] for i in range(k)]  # clusters will be that way [[(x1,y1),..,(xn,yn)],...[(x1, y1), ...,(xm,ym)]].
    centers = get_start_centers(k, x, y)  # random  POINT
    prev_centers = centers[:]
    is_continue = True

    while is_continue:
        clusters = [[] for i in range(k)]  # we must renew our clusters every iteration

        for i in range(len(X)):
            min_distance = float('inf')  # +infinity
            index_of_minimal = 0
            for j in range(len(centers)):  # decide which cluster
                euclid_distance = calc_euclid(x[i], y[i], centers[j][0], centers[j][1])
                if euclid_distance < min_distance:
                    min_distance = euclid_distance
                    index_of_minimal = j
            clusters[index_of_minimal].append((x[i], y[i]))  # append a cortege

        for i in range(len(prev_centers)):
            prev_centers[i] = list(centers[i])  # we must copy this way.

        for i in range(len(centers)):  # calculating new centers
            sx = 0
            sy = 0
            for j in range(len(clusters[i])):
                sx += clusters[i][j][0]
                sy += clusters[i][j][1]
            if len(clusters[i]) > 0:  # situation when cluster is empty is possible
                centers[i][0] = round(sx / len(clusters[i]), 5)
                centers[i][1] = round(sy / len(clusters[i]), 5)

        is_continue = False
        for i in range(len(centers)):  # decide must we continue or not
            if centers[i] not in prev_centers:
                is_continue = True
                break

    return clusters


X = read_column_from_csv(0 + (v_number - 1) * 2, 'data/7problem.csv')
Y = read_column_from_csv(1 + (v_number - 1) * 2, 'data/7problem.csv')
draw_plot(X, Y)
for k in range(2, 5):
    clusters = clustering(k, X, Y)
    draw_clusters(clusters)
