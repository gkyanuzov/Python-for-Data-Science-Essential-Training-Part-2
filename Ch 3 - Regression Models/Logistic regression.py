import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

sb.set_style('whitegrid')
rcParams['figure.figsize'] = 5, 4

""""

Logistic regression on titanic dataset

"""

address = 'D:\\Docs\\Uni\\Online Courses\\Python for Data Science Essential Training Part 2 - ' \
          'LinkedInLearning\\Ex_Files_Python_Data_Science_EssT_Pt2\\Exercise Files\\Data\\titanic-training-data.csv'

titanic_training = pd.read_csv(address)
titanic_training.columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
                            'Fare', 'Cabin', 'Embarked']
print(titanic_training.head(10))

print(titanic_training.info())

# check if target variable is binary
print(sb.countplot(x='Survived', data=titanic_training, palette='hls'))
# plt.show()
print('\n')

# check for missing values
print(titanic_training.isnull().sum())
print('\n')
print(titanic_training.describe())
print('\n')

# drop irrelevant values - in this case -  name, ticket number, passenger id; cabin number is almost all missing
# values so we drop it as well

titanic_data = titanic_training.drop(['Name', 'Ticket', 'Cabin'], axis=1)
print(titanic_data.head(3))
print('\n')

# imputing missing values - in this case age variable
sb.boxplot(x='Parch', y='Age', data=titanic_data, palette='hls')
# plt.show()

# using parch variable
parch_groups = titanic_data.groupby(titanic_data['Parch'])
print(parch_groups.mean(numeric_only=True))


def age_approx(cols):
    age = cols[0]
    parch = cols[1]

    if pd.isnull(age):
        if parch == 0:
            return 32
        elif parch == 1:
            return 24
        elif parch == 2:
            return 17
        elif parch == 3:
            return 33
        elif parch == 4:
            return 45
        else:
            return 30  # mean age in df
    else:
        return age


# apply the function
titanic_data['Age'] = titanic_data[['Age', 'Parch']].apply(age_approx, axis=1)
print(titanic_data.isnull().sum())

# embarked col has only 2 na values out of 891, so we can drop them
titanic_data.dropna(inplace=True)
titanic_data.reset_index(inplace=True, drop=True)
print(titanic_data.info())

# converting categorical variables to dummy indicators
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# convert gender to binary - 1 = male, 0 = female
gender_cat = titanic_data['Sex']
gender_encoded = label_encoder.fit_transform(gender_cat)
print(gender_encoded[0:5])

gender_df = pd.DataFrame(gender_encoded, columns=['male_gender'])
print(gender_df.head())

# convert embarked to binary
embarked_cat = titanic_data['Embarked']
embarked_encoded = label_encoder.fit_transform(embarked_cat)
print(embarked_encoded[0:100])
# embarked has 3 values, so we create a binary column  for each one
from sklearn.preprocessing import OneHotEncoder
binary_encoder = OneHotEncoder(categories='auto')
embarked_onehot = binary_encoder.fit_transform(embarked_encoded.reshape(-1, 1))
embarked_onehot_mat  = embarked_onehot.toarray()
embarked_df = pd.DataFrame(embarked_onehot_mat, columns=['C', 'Q', 'S'])
print(embarked_df.head())

# drop gender and embarked from og df
titanic_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)
print(titanic_data.head())

# add the new cols we made to the df
titanic_dmy = pd.concat([titanic_data, gender_df, embarked_df], axis = 1, verify_integrity= True).astype(float) #axis=1 means we add them as columns
print(titanic_dmy.head())

# check for independence between features
sb.heatmap(titanic_dmy.corr())
# plt.show()

# pclass and fare are dependant, so we drop them
titanic_dmy.drop(['Fare', 'Pclass'], axis=1, inplace=True)
print(titanic_dmy.head())

# check if dataset size is sufficient, at least 50 records per predictive feature
print(titanic_dmy.info())

# train and test set
x_train, x_text, y_train,y_test = train_test_split(titanic_dmy.drop('Survived', axis=1),
                                                   titanic_dmy['Survived'], test_size=0.2,
                                                   random_state=200)
print(x_train.shape)
print(y_train.shape)
print(x_train[0:5])

# deploying and evaluating the model
logreg = LogisticRegression(solver='liblinear')
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_text)

# classifciation report without cross-validation
print(classification_report(y_test, y_pred))

# k-fold cross-validation & confusion matrices
y_train_pred = cross_val_predict(logreg, x_train, y_train, cv=5)
print(confusion_matrix(y_train, y_train_pred)) # first row - correct predictions, second row - incorrect predictions
print(precision_score(y_train, y_train_pred))

# make a test prediction
print(titanic_dmy[863:864])
test_passenger = np.array([866, 40, 0, 0,0,0,0,1]).reshape(1, -1)
print(logreg.predict(test_passenger))
print(logreg.predict_proba(test_passenger))
# predicts 1, so this test pass will survive, 73 % chance of correct prediction