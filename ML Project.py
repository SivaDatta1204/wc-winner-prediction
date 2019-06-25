# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:55:25 2019

@author: SIVA DATTA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=20,random_state=0)


#importing the data set
world_cup=pd.read_csv('World Cup 2019 Dataset.csv')
results=pd.read_csv('results.csv')
#results of matchess played only by india
df = results[(results['Team_1'] == 'India') | (results['Team_2'] == 'India')]
india = df.iloc[:]
#teams participating this year
worldcup_teams = ['England', ' South Africa', '', 'West Indies', 
            'Pakistan', 'New Zealand', 'Sri Lanka', 'Afghanistan', 
            'Australia', 'Bangladesh', 'India']
df_teams_1 = results[results['Team_1'].isin(worldcup_teams)]
df_teams_2 = results[results['Team_2'].isin(worldcup_teams)]
df_teams = pd.concat((df_teams_1, df_teams_2))
df_teams.drop_duplicates()
df_teams.count()
#deleting the un-necessary data
df_used=df_teams.drop(['date','Margin','Ground'],axis=1)
#arranging the index indices
df_used= df_used.reset_index(drop=True)
#encoding to continuous variables
#final = pd.get_dummies(df_used, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2',])
#creating the data sets
X = df_used.drop(['Winner'], axis=1)
y = df_used["Winner"]
#one hot encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.values[:, 0] = labelencoder_X.fit_transform(X.values[:, 0])
X.values[:, 1] = labelencoder_X.fit_transform(X.values[:, 1])
#X.values[:, 2] = labelencoder_X.fit_transform(X.values[:, 2])
##X.values[:, 3] = labelencoder_X.fit_transform(X.values[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#encoding y values
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#creating the train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fitting decision tree classifier
#########from sklearn.tree import DecisionTreeClassifier
#########classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
#########classifier.fit(X_train,y_train)
#########y_pred = classifier.predict(X_test)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

#print("Training set accuracy: ", '%.3f'%(y_pred))
#creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Applying the k  folld cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=rf,X=X_test,y=y_pred,cv=10)
accuracies.mean()
# Loading new datasets
ranking = pd.read_csv('icc_rankings.csv') 
fixtures = pd.read_csv('fixtures.csv')

# List for storing the group stage games
pred_set = []

# Create new columns with ranking position of each team
fixtures.insert(1, 'first_position', fixtures['Team_1'].map(ranking.set_index('Team')['Position']))
fixtures.insert(2, 'second_position', fixtures['Team_2'].map(ranking.set_index('Team')['Position']))
#classifyng the data
fixtures = fixtures.iloc[:45, :]
fixtures.tail()
# Loop to add teams to new prediction dataset based on the ranking position of each team
for index, row in fixtures.iterrows():
    if row['first_position'] < row['second_position']:
        pred_set.append({'Team_1': row['Team_1'], 'Team_2': row['Team_2'], 'winning_team': None})
    else:
        pred_set.append({'Team_1': row['Team_2'], 'Team_2': row['Team_1'], 'winning_team': None})
pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set
pred_set.head()
# Add missing columns compared to the model's training dataset
missing_cols = set(df_used.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[df_used.columns]
# Get dummy variables and drop winning_team column
pred_set = pd.get_dummies(pred_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])
pred_set = pred_set.drop(['Winner','Ground'], axis=1)
pred_set.head()
#group matches 
predictions = rf.predict(pred_set.values.reshape(-1,1))
for i in range(fixtures.shape[0]):
    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
    if predictions[i] == 1:
        print("Winner: " + backup_pred_set.iloc[i, 1])
    
    else:
        print("Winner: " + backup_pred_set.iloc[i, 0])
    print("")