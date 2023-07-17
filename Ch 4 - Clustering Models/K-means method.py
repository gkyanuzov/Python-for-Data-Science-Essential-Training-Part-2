#usuperviessed ml algorithm
# scale variables, look at a scatterplot beforehand
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import scale
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

plt.figure(figsize=(7,4))
iris = datasets.load_iris()
x = scale(iris.data)
y = pd.DataFrame(iris.target)
variable_names = iris.feature_names
print(x[0:10])


# building and running the model

clustering = KMeans(n_clusters=3, random_state=5)
clustering.fit(x)

# plotting the model outputs
iris_df = pd.DataFrame(iris.data)
iris_df.columns = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width']
y.columns = ['Targets']

color_theme = np.array(['darkgrey', 'lightsalmon', 'powderblue'])

plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_length, y=iris_df.Petal_width, c= color_theme[iris.target], s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_length, y=iris_df.Petal_width, c= color_theme[clustering.labels_], s=50)
plt.title('K-Means Classification')
plt.show()

# relabeling
relabel = np.choose(clustering.labels_, [2,0,1]).astype(np.int64)

plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_length, y=iris_df.Petal_width, c= color_theme[iris.target], s=50)
plt.title('Ground Truth Classification')

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_length, y=iris_df.Petal_width, c= color_theme[relabel], s=50)
plt.title('K-Means Classification')
plt.show()

# evaluate the clustering results
print(classification_report(y, relabel))

# high precision + high recall = highly accurate model results
