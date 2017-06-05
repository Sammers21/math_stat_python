import pandas as pd
import statsmodels.formula.api as smf
import math
from lib import rss, ess
from scipy.stats import f

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

# Прогноз выживания по модели
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

ess_y = ess(survived, y_estimate)
rss_y = rss(survived, y_estimate)
k = 4  # кол-во коэффициентов (с учётом свободного)
n = len(survived)  # объём выборки

f_crit = f.ppf(0.95, k - 1, n - k)
f_real = ess_y / (k - 1) / (rss_y / (n - k))

if f_crit < f_real:
    print('Отвергаем гипотзу о значимости модели регрессии в целом H0:(b1=b2=b3=b4=0)')
else:
    print('Принмаем гипотзу о значимости модели регрессии в целом H0:(b1=b2=b3=b4=0)')

# В данном случае, показателем того, на сколько модель расходится с реальным положенимем дел
# является RSS. В данном случаем значение RSS - колчество наблюдений, где модель расхоидтся с данными.
print('Модель logit расходится с данными в {} случаях из {}'.format(rss_y, n))
