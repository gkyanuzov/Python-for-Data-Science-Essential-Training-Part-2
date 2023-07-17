# unsupervised method that cluster core samples(dense areas of a dataset)
# and denotes non-core samples(sparse portions of the dataset)
# use this method to identify collective outliers
#  should make up <= 5 % of the total observations
# important dbscan model parameters:
# eps, min_samples

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb
import sklearn
from sklearn.cluster import DBSCAN
from collections import Counter

rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')

address = 'D:\\Docs\\Uni\\Online Courses\\Python for Data Science Essential Training Part 2 - ' \
          'LinkedInLearning\\Ex_Files_Python_Data_Science_EssT_Pt2\\Exercise Files\\Data\\iris.data.csv'

df = pd.read_csv(address, header=None, sep=',')
df.columns = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width', 'Species']

data = df.iloc[:, 0:4].values
target = df.iloc[:, 4].values  # select column at index 4 - Species
print(df.head())

# eps - max distance between one neightbourhood points - 0.8
model = DBSCAN(eps=0.8, min_samples=19).fit(data)
print(model)

outliers_df = pd.DataFrame(data)
print(Counter(model.labels_))
print(outliers_df[model.labels_ == -1])

fig = plt.figure()
ax = fig.add_axes([.1, .1, 1, 1])
colors = model.labels_
ax.scatter(data[:,2], data[:,1], c=colors, s= 120)
ax.set_xlabel('petal length')
ax.set_ylabel('sepal with')
plt.title('DBSCAN for outlier detection')

plt.show()