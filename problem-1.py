from functools import reduce

from scipy.stats import rv_continuous
import numpy as np


class customDist(rv_continuous):
    """"
    Distribution F(x)=1-exp(-exp(0.1x)
    """
    #TODO поменять тут на своё распределение
    def _cdf(self, x, *args):
        return 1 - np.exp(-np.exp(x / 10.))


d = customDist()


#Генерируем выборку из 100 случайных велечи
sample = d.rvs(size=100)

import matplotlib.pyplot as m

f = m.figure()
m.hist(sample, bins=10)
m.title("Distribution F(x)=1-exp(-exp(0.1x)) 100xValues")
m.xlabel("Value")
m.ylabel("Count of times")
m.grid(True)

#Генерируем выборку из 1000 случайных величин
sample = d.rvs(size=1000)

f = m.figure()
m.hist(sample, bins=10)
m.title("Distribution F(x)=1-exp(-exp(0.1x)) 1000xValues")
m.xlabel("Value")
m.ylabel("Count of times")
m.grid(True)

#Генерируем выборку из 1000 случайных велечин
#где каждая случайная величина является суммой 30 случайных велечин из исходной выборки
sample = [sum(d.rvs(size=30)) for i in range(1000)]

f = m.figure()
m.hist(sample, bins=10)
m.title("Distribution Yi=SUM j=1...30 Xij i=1...30 1000xValues")
m.xlabel("Value")
m.ylabel("Count of times")
m.grid(True)

m.show()
