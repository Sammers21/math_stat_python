from functools import reduce

from scipy.stats import rv_continuous
import numpy as np


class customDist(rv_continuous):
    """"
    Distribution F(x)=1-exp(-exp(0.1x)
    """

    def _cdf(self, x, *args):
        return 1 - np.exp(-np.exp(x / 10.))


d = customDist(name='gaussian')
x = d.rvs()

sample = []
for i in range(100):
    sample.append(d.rvs())

import matplotlib.pyplot as m

f = m.figure()
m.hist(sample, bins=10)
m.title("Distribution F(x)=1-exp(-exp(0.1x)) 100xValues")
m.xlabel("Value")
m.ylabel("Count of times")
m.grid(True)

sample = []
for i in range(1000):
    sample.append(d.rvs())

f = m.figure()
m.hist(sample, bins=10)
m.title("Distribution F(x)=1-exp(-exp(0.1x)) 1000xValues")
m.xlabel("Value")
m.ylabel("Count of times")
m.grid(True)

sample = []
for i in range(1000):
    lst = []
    for y in range(30):
        lst.append(d.rvs())
    sample.append(reduce(lambda x, y: x + y, lst))

f = m.figure()
m.hist(sample, bins=10)
m.title("Distribution Yi=SUM j=1...30 Xij i=1...30 1000xValues")
m.xlabel("Value")
m.ylabel("Count of times")
m.grid(True)

m.show()
