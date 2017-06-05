import pandas as pd
import statsmodels.formula.api as smf
import math

from lib import mk_data_var, read_column_from_csv

# TODO alter this to your variant
v_number = 1

mk_data_var(v_number)

class_1 = read_column_from_csv(0, 'data/6problem_{}.csv'.format(v_number), type='f')
class_2 = read_column_from_csv(1, 'data/6problem_{}.csv'.format(v_number), type='f')
class_3 = read_column_from_csv(2, 'data/6problem_{}.csv'.format(v_number), type='f')
sex = read_column_from_csv(3, 'data/6problem_{}.csv'.format(v_number), type='f')
survived = read_column_from_csv(4, 'data/6problem_{}.csv'.format(v_number), type='f')

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
for i in range(len(res.params)):
    print(res.params[i])

y_estimate = [1
              if
              math.exp(
                  res.params[0] +
                  res.params[1] * sex[i] +
                  res.params[2] * class_1[i] +
                  res.params[3] * class_2[i] +
                  res.params[4] * class_3[i]) > 1
              else 0

              for i in range(len(sex))]

print(y_estimate)

"""
# ESS=ess(su)
# RSS=rss()

model = smf.logit(formula="survived ~ 1 + C(sex) + C(class_1) + C(class_2) + C(class_3)", data=df)
res = model.fit()
print(res.summary())

model = smf.ols(formula="survived ~ 1 + sex + class_1 + class_2 + class_3", data=df)
res = model.fit()
print(res.summary())

model = smf.logit(formula="survived ~ 1 + C(sex) + C(class_1) + C(class_2) + C(sex * class_1) + C(sex * class_2)",
                  data=df)
res = model.fit()
print(res.summary())
"""
