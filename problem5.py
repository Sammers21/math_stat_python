import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from lib import read_column_from_csv


def draw_scatter(title, sample_1, title_1, sample_2, title_2):
    plt.title(title)
    plt.xlabel(title_1)
    plt.ylabel(title_2)

    plt.scatter(sample_1, sample_2)
    plt.show()


def draw_histogram(title, sample, bins):
    plt.title(title)
    plt.hist(sample, bins=bins)
    plt.show()

"""
Немного теории:
    Нужно придумать такую модель которая наилучшим образом ложится на предоставленные данные.
    Универсальным показателем того на сколько удачно является RSS(Residual sum of squares) 
    или сумма квадратов остатков. Т.е. если RSS модели относительно небольшой - значит модель хорошоая.
    


Характериристики:
    totsp общая площадь квартиры (в кв.м);
    price цена квартиры в долларах США;
    d2=1, если квартира двухкомнатная, 0 иначе;
    d3=1, если квартира трёхкомнатная, 0 иначе;
    d4=1, если квартира четырёхкомнатная, 0 иначе;
    dist расстояние от центра Москвы (в км);
    walk=1,  если до метро можно быстро дойти пешком, 0 иначе;
    brick=1,  если дом кирпичный, 0 иначе;
    bal=1,  если есть балкон, 0 иначе;
    floor=0,  если этаж первый или последний, 1 иначе.
"""

variation = 1

# TODO Вы должны сделать свой .csv файл из того что прислал Фурманов. Нужно удалить из него все строки
# в которых есть пустые элементы. Пустые строки в каждом варианте разные, поэтому удалите
# только те, что пустые именно в вашем варианте. После этих операций сохраните его как data/5problem.csv

bal = read_column_from_csv(column_number=0 + (variation - 1) * 11, file='data/5problem.csv')
brick = read_column_from_csv(column_number=1 + (variation - 1) * 11, file='data/5problem.csv')
d2 = read_column_from_csv(column_number=2 + (variation - 1) * 11, file='data/5problem.csv')
d3 = read_column_from_csv(column_number=3 + (variation - 1) * 11, file='data/5problem.csv')
d4 = read_column_from_csv(column_number=4 + (variation - 1) * 11, file='data/5problem.csv')
dist = read_column_from_csv(column_number=5 + (variation - 1) * 11, file='data/5problem.csv')
floor = read_column_from_csv(column_number=6 + (variation - 1) * 11, file='data/5problem.csv')
price = read_column_from_csv(column_number=7 + (variation - 1) * 11, file='data/5problem.csv')
totsp = read_column_from_csv(column_number=8 + (variation - 1) * 11, file='data/5problem.csv')
walk = read_column_from_csv(column_number=9 + (variation - 1) * 11, file='data/5problem.csv')

print(bal)

df = pd.DataFrame({
    "price": price,
    "walk": walk,
    "totsp": totsp,
    "floor": floor,
    "dist": dist,
    "d2": d2,
    "d3": d3,
    "d4": d4,
    "brick": brick,
    "bal": bal
})
n = len(price)

# about OLS - Ordinary least squares
# In statistics, ordinary least squares (OLS) or linear least squares is a method
# for estimating the unknown parameters in a linear regression model, with the goal
# of minimizing the sum of the squares of the differences between the observed responses
# (values of the variable being predicted) in the given dataset and those predicted by a
# linear function of a set of explanatory variables. Visually this is seen as the sum of
# the squared vertical distances between each data point in the set and the corresponding point
# on the regression line – the smaller the differences, the better the model fits the data.
# The resulting estimator can be expressed by a simple formula, especially in the case of
# a single regressor on the right-hand side.
# https://en.wikipedia.org/wiki/Ordinary_least_squares

#TODO Выбрать модель
# Для этого нужно прочесть: https://github.com/bdemeshev/epsilon/raw/master/e_001/functional-form/functional-form.pdf

# Модель номер 1 - Линейная
lin_mod = smf.ols(formula="price ~ totsp + dist + walk + d2 + d3 + d4 + brick + bal + floor", data=df)
lin_res = lin_mod.fit()
print("RSS =", sum(np.square(lin_res.resid)))
print(lin_res.summary())
draw_histogram("Lear regression (ost)", lin_res.resid, 100)
draw_scatter("Lear regression", lin_res.fittedvalues, "fitted_price", lin_res.resid, "price - fitted_price")


# Модель номер 2 - ПолуЛогорифмическая
l_h_mod = smf.ols(formula="np.log(price)~ totsp + dist + walk + d2 + d3 + d4 + brick + bal + floor", data=df)
l_h_res = l_h_mod.fit()
print("RSS =", sum(np.square(l_h_res.resid)))
print(l_h_res.summary())
draw_histogram("Log_h (ost)", l_h_res.resid, 20)
draw_scatter("Log_h res", l_h_res.fittedvalues, "log(fitted_price)", l_h_res.resid, "log(price) - log(fitted_price)")

# Модель номер 3 - Логорифмическая
l_mod = smf.ols(formula="np.log(price)~ np.log(totsp) + dist + walk + d2 + d3 + d4 + brick + bal + floor", data=df)
l_res = l_mod.fit()
print("RSS =", sum(np.square(l_res.resid)))
print(l_res.summary())
draw_histogram("Log (ost)", l_res.resid, 20)
draw_scatter("Log res", l_res.fittedvalues, "log(fitted_price)", l_res.resid, "log(price) - log(fitted_price)")

"""
#TODO После того как вы нашли хорошие коэфициенты

Рассчитайте прогноз цены однокомнатной квартиры с указанными характеристиками:
    площадь 40 кв.м.,
    есть балконом,
    кирпичный девятиэтажный дом,
    третий этаж,
    в 10 км от центра Москвы, 
    рядом с метро
"""