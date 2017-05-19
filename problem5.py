from lib import read_column_from_csv

"""
totsp
общая площадь квартиры (в кв.м);
price
цена квартиры в долларах США;

d2
=1, если квартира двухкомнатная, 0 иначе;

d3
=1, если квартира трёхкомнатная, 0 иначе;

d4
=1, если квартира четырёхкомнатная, 0 иначе;

dist
расстояние от центра Москвы (в км);

walk
=1,  если до метро можно быстро дойти пешком, 0 иначе;

brick
=1,  если дом кирпичный, 0 иначе;

bal
=1,  если есть балкон, 0 иначе;

floor
=0,  если этаж первый или последний, 1 иначе.

"""

variation = 1

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
