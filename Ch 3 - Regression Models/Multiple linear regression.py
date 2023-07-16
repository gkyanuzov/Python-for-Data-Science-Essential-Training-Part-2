import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import seaborn as sb
from collections import Counter

sb.set_style('whitegrid')
rcParams['figure.figsize'] = 5,4

address = 'D:\\Docs\\Uni\\Online Courses\\Python for Data Science Essential Training Part 2 - ' \
          'LinkedInLearning\\Ex_Files_Python_Data_Science_EssT_Pt2\\Exercise Files\\Data\\enrollment_forecast.csv'
enroll = pd.read_csv(address)
enroll.columns = ['year', 'roll', 'unem', 'hgrad', 'inc']
print(enroll.head())


# sb.pairplot(enroll)
plt.show()
print(enroll.corr())

enroll_data = enroll[['unem', 'hgrad']].values
enroll_target = enroll[['roll']].values
enroll_data_names = ['unem', 'hgrad']

x , y = scale(enroll_data), enroll_target

# checking for missing values
missing_values = x == np.NAN
print(x[missing_values == True])

linreg = LinearRegression()
linreg.fit(x,y)
print(linreg.score(x,y))