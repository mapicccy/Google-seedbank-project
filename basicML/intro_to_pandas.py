from __future__ import print_function

import pandas as pd
import numpy as np
print(pd.__version__)

city_names = pd.Series(['San Francisico', 'San Jose', 'Sacramento'])
population = pd.Series([100001, 210000, 3432111])
df = pd.DataFrame({'City name': city_names, 'Population': population})
print(df)
# print(type(df['City name']))
# print(df['City name'][1])
# print(type(df['City name'][1]))
# print(population/1000)
# print(np.log(population))
print(city_names.index, population.index)
print(city_names.reindex([2, 0, 1]))

california_housing_dataframe = pd.read_csv('https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv', sep=',')
california_housing_dataframe.describe()
california_housing_dataframe.hist('housing_median_age')