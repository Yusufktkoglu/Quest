# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:13:03 2024

@author: 90539
"""


#Visualize the data (so that people like me can understand!)
#Clean up the data (balance it out, impute missing values and so on… depending on the method you are going to use!)
#Visualize the cleaned data (so that people like me can understand the effect of cleaning process!)
#Create a simple model that performs reasonably well. (If it doesn’t perform well, comment on why and how to improve it!)
#Evaluate the model with a testset you will create from the dataset. (Pretty plots make things easier to understand)
#Upload your code to a private github repo you can share with us, and invite us (https://github.com/alpsina, https://github.com/ltc0060 and https://github.com/ahmetkoklu) as collaborators so only we can see our super-secret project.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
df=pd.read_csv("dataset.csv")

feature_1 = df['feature_1']
feature_2 = df['feature_2']
feature_3 = df['feature_3']
feature_4 = df['feature_4']
is_virus = df['isVirus']
data = pd.DataFrame({'Feature 1': feature_1,
                     'Feature 2': feature_2,
                     'Feature 3': feature_3,
                     'Feature 4': feature_4})
sns.pairplot(data)
plt.show()


feature_1.fillna(feature_1.mean(),inplace=True)
feature_2.fillna(feature_2.mean(),inplace=True)
feature_3.fillna(feature_3.mean(),inplace=True)
feature_4.fillna(feature_4.mean(),inplace=True)
data1 = pd.DataFrame({'Feature 1': feature_1,
                     'Feature 2': feature_2,
                     'Feature 3': feature_3,
                     'Feature 4': feature_4})

sns.pairplot(data1)
plt.show()

X=df[['feature_1','feature_2','feature_3','feature_4']]
y=df['isVirus']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(classification_report(y_test, y_pred))





