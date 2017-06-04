from lib import mk_data_var, read_column_from_csv, ess, rss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from functools import reduce

# TODO alter this to your variant
v_number = 1

mk_data_var(v_number)

class_1 = read_column_from_csv(0, 'data/6problem_{}.csv'.format(v_number), type='f')
class_2 = read_column_from_csv(1, 'data/6problem_{}.csv'.format(v_number), type='f')
class_3 = read_column_from_csv(2, 'data/6problem_{}.csv'.format(v_number), type='f')
sex = read_column_from_csv(3, 'data/6problem_{}.csv'.format(v_number), type='f')
survived = read_column_from_csv(4, 'data/6problem_{}.csv'.format(v_number), type='f')

# survived  overall count
#       \   /
c1_s_1 = [0, 0]
c1_s_0 = [0, 0]
c2_s_1 = [0, 0]
c2_s_0 = [0, 0]
c3_s_1 = [0, 0]
c3_s_0 = [0, 0]

# govno
for i in range(len(class_1)):

    if class_1[i] == 1:
        if sex[i] == 1:
            c1_s_1[1] += 1
            c1_s_1[0] += survived[i]
        else:
            c1_s_0[1] += 1
            c1_s_0[0] += survived[i]

    elif class_2[i] == 1:
        if sex[i] == 1:
            c2_s_1[1] += 1
            c2_s_1[0] += survived[i]
        else:
            c2_s_0[1] += 1
            c2_s_0[0] += survived[i]

    elif class_3[i] == 1:
        if sex[i] == 1:
            c3_s_1[1] += 1
            c3_s_1[0] += survived[i]
        else:
            c3_s_0[1] += 1
            c3_s_0[0] += survived[i]

print(c1_s_1)
print(c1_s_0)
print(c2_s_1)
print(c2_s_0)
print(c3_s_1)
print(c3_s_0)

chances = []
for i in range(len(class_1)):

    if class_1[i] == 1:
        if sex[i] == 1:
            chances.append((c1_s_1[0] / c1_s_1[1]) / (1 - c1_s_1[0] / c1_s_1[1]))
        else:
            chances.append((c1_s_0[0] / c1_s_0[1]) / (1 - c1_s_0[0] / c1_s_0[1]))
    elif class_2[i] == 1:
        if sex[i] == 1:
            chances.append((c2_s_1[0] / c2_s_1[1]) / (1 - c2_s_1[0] / c2_s_1[1]))
        else:
            chances.append((c2_s_0[0] / c2_s_0[1]) / (1 - c2_s_0[0] / c2_s_0[1]))

    elif class_3[i] == 1:
        if sex[i] == 1:
            chances.append((c3_s_1[0] / c3_s_1[1]) / (1 - c3_s_1[0] / c3_s_1[1]))
        else:
            chances.append((c3_s_0[0] / c3_s_0[1]) / (1 - c3_s_0[0] / c3_s_0[1]))

df = pd.DataFrame({
    "class_1": class_1,
    "class_2": class_2,
    "class_3": class_3,
    "sex": sex,
    "survived": survived
})

# Оцените модель логит для вероятности выжить в зависимости от пола, возраста и класса каюты:
model = smf.logit(formula="survived ~ 1 + C(sex) + C(class_1) + C(class_2) + C(class_3)", data=df)
res = model.fit()
print(res.summary())

# Проверьте значимость модели в целом
print(res.params[0])
print(res.params[0])
print(res.params[0])


#ESS=ess(su)
#RSS=rss()

model = smf.probit(formula="survived ~ 1 + C(sex) + C(class_1) + C(class_2) + C(class_3)", data=df)
res = model.fit()
print(res.summary())

model = smf.ols(formula="survived ~ 1+ C(sex) + C(class_1) + C(class_2) + C(class_3)", data=df)
res = model.fit()
print(res.summary())

model = smf.logit(formula="survived ~ 1 + C(sex) + C(class_1) + C(class_2) + C(sex * class_1) + C(sex * class_2)", data=df)
res = model.fit()
print(res.summary())
