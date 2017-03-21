[![Build Status](https://travis-ci.org/Sammers21/math_stat_python.svg?branch=master)](https://travis-ci.org/Sammers21/math_stat_python)

#Вопросы к заданию номер 1

#####1
**Вопрос:** Как можно сгенерировать ваше распределение с использованием лишь равномерного распреления R(0,1)?  
**Ответ:** находим обратную функцию и подставляем значения равномерного распределения.

#####2
**Вопрос** (вытекающий из первого): А какая обратная функция к вашей?  
**Ответ:** взял мел и написал её на доске или бумаге

#####3
**Вопрос:** что такое медиана и какова медиана вашего распределения?  
**Ответ:** медиана - квантиль уровня 0.5, т.е. такое значение распределения, получить значение меньше которого можно с вероятностью 1/2. Чтобы найти его, нужно режить уравнения вида F(x) = 1/2, где F - функция распределения.

#####4
**Вопрос:** дайте определение центральной предельной теоремы (ЦПТ)  
**Ответ:**
![2017-03-09_20-37-36](https://cloud.githubusercontent.com/assets/8942211/23763400/8225d030-050a-11e7-87fc-80bf28ab529e.jpg)

#####5
**Вопрос:** что такое дисперсия?  
**Ответ:** дисперсия: D(X) = E((X-E(X))^2)  

#####6
**Вопрос:** Какие величины называют независимыми?  
**Ответ:** Случайные величины называются независимыми, если закон распределения каждой из них не зависит от того, какое значение приняла другая. В противном случае величины и называются зависимыми. Для независимых величин выполняется:
1. P(X ∈ A, Y ∈ B) = P(X ∈ A) * P (Y ∈ B)
2. f(x, y) = f1(x)*f2(y)
3. F(x, y) = F1(x)*F2(y)
4. f(y|x) = f2(y)


#Вопросы к заданию номер 2

#####1
**Вопрос:** Что такое ошибка первого и второго рода?  
**Ответ:**  
![screenshot from 2017-03-05 21-20-44](https://cloud.githubusercontent.com/assets/16746106/23590054/1ec50b28-01ea-11e7-93da-3511d45e1e24.png)

#####2
**Вопрос:** что такое уровень доверия?  
**Ответ:** Уровень доверия - статистический термин, означающий вероятность того, что доверительный интервал содержит истинное значение параметра.

#####3
**Вопрос:** а какую вы будете использовать статистику для оценки:

			a) мат ожидания с известной дисперсией
			б) мат ожидания с неизвестной дисперсией
			в) дисперсии с известным мат ожиданем
			г) дисперсии с неизвестным мат ожиданем
	
**Ответы на a) и б)**  
![image](https://cloud.githubusercontent.com/assets/16746106/23589495/42427c94-01df-11e7-8291-6169fdc557a0.png)

**Ответы на в) и г)**  
![screenshot from 2017-03-05 20-03-24](https://cloud.githubusercontent.com/assets/16746106/23589484/0e09ba3c-01df-11e7-934a-f6787ce6a1ea.png)

#####4
**Вопрос:** Напишите формулу для t-статистики?  
**Ответ:** 	![image](https://wikimedia.org/api/rest_v1/media/math/render/svg/f3ca1b501196490641280f77e29568f438be79d1)  

###Вклад
#####Каждый из вас, кто читает это README может помочь своим однокурсникам.
- Если тут нет вопроса, который Фурманов задавал вам, то не стесняйтесь и добавьте его (посредством pull request),
- Если у вас есть проблема, с которой вы столкнулись и не можете решить, то создайте issue в этом репозитории. Помощь обязательно будет. Быстрая и оперативная.
- Если вы считаете, что в коде, который демонстрирует примерное решение задачи, есть ошибка, то непременно исправьте её или сообщите о ней.

