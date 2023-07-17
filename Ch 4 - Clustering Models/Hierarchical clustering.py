# unsupervised ml, predict subgroups by findings the distance
# between each data point and its nearest neighbours, and then linking
# the most nearby neighbours
# use cases : business process management, customer segmentation, social network analysis
# parameters:
# distance metrics - euclidean, manhattan, cosine
# linkage parameters - ward, complete, average

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb
import sklearn
import sklearn.metrics as sm
from sklearn.cluster import AgglomerativeClustering
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist

np.set_printoptions(precision=4, suppress=True)
plt.figure(figsize=(10,3))
plt.style.use('seaborn-whitegrid')


address = 'D:\\Docs\\Uni\\Online Courses\\Python for Data Science Essential Training Part 2 - ' \
          'LinkedInLearning\\Ex_Files_Python_Data_Science_EssT_Pt2\\Exercise Files\\Data\\mtcars.csv'

cars = pd.read_csv(address)
cars.columns = ['car_names','mpg','cyl','disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

x = cars[['mpg', 'disp', 'hp', 'wt']].values

y = cars.iloc[:,(9)].values # select col number 9

# use scipy to generate dendograms
z = linkage(x, 'ward')
dendrogram(z, truncate_mode='lastp', p=12, leaf_rotation= 45, leaf_font_size=15, show_contracted=True)
plt.title('Trunceted hierarchical clustering diagram')
plt.xlabel('Cluster size')
plt.ylabel('Distance')

plt.axhline(y=500)
plt.axhline(y=150)
plt.show()

# generating hierarchical clusters
k = 2
hClustering = AgglomerativeClustering(n_clusters=k, affinity= 'euclidean', linkage='ward')
hClustering.fit(x)
print(sm.accuracy_score(y, hClustering.labels_))

hClustering = AgglomerativeClustering(n_clusters=k, affinity= 'euclidean', linkage='average')
hClustering.fit(x)
print(sm.accuracy_score(y, hClustering.labels_))

hClustering = AgglomerativeClustering(n_clusters=k, affinity= 'manhattan', linkage='average')
hClustering.fit(x)
print(sm.accuracy_score(y, hClustering.labels_))

