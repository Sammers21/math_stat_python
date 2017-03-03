from functools import reduce

import numpy.random as l
import numpy as n
import matplotlib.pyplot as p

mean = 3
sigma = 1
count_of_elements = 8
# квантиль уровня P (X<x)=0.4750  для N(0,1)
cvantil0025gaussian = 1.96


def test(m=mean, c=count_of_elements):
    global h_one_true, h_two_true
    # of course sum is 10000
    h_one_true = 0
    h_two_true = 0
    for i in range(10000):
        array_of_elements_from_distribution = l.normal(m, sigma, c)
        average = reduce(lambda x, y: x + y, array_of_elements_from_distribution) / c
        right = average + (cvantil0025gaussian * sigma / n.sqrt(c))
        left = average - (cvantil0025gaussian * sigma / n.sqrt(c))
        if (left < mean and mean < right):
            h_one_true += 1
        else:
            h_two_true += 1
    return (h_one_true, h_two_true)


test()

print("result is :\n"
      "hypnosis than mean is 3 was true {0} times\n"
      "hypnosis than mean isn't 3 was true {1} times\n"
      "оценка вероятности ошибки первого рода {2} "
      .format(h_one_true, h_two_true, h_two_true / 10000)
      )
"""
result is :
hypnosis than mean is 3 was true 9454 times
hypnosis than mean isn't 3 was true 546 times
оценка вероятности ошибки первого рода 0.0546
"""

counter = 1
array_of_points = []
array_of_counters = []
for i in range(9):
    test(counter)
    array_of_points.append(h_two_true / 10000)
    array_of_counters.append(counter)
    counter += 0.5

p.plot(array_of_counters, array_of_points,
       linewidth=3,
       color='r')

counter = 1
array_of_points = []
array_of_counters = []
for i in range(9):
    test(counter, 50)
    array_of_points.append(h_two_true / 10000)
    array_of_counters.append(counter)
    counter += 0.5

p.plot(array_of_counters, array_of_points,
       linewidth=3,
       color='g')

p.xlabel("Мат ожидание генерируемой выборки(настоящее)")
p.ylabel("Вероятость отвергнуть основную гипотезу")

p.text(2.3, 0.9, "Выборка из 50 элементов", rotation=300,color='g')
p.text(1, 0.85, "Выборка из 8 элементов", rotation=300,color='r')



p.show()
