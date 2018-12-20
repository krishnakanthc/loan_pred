# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:51:57 2018

@author: user
"""

import pandas as pd
import numpy as np
import os
from fancyimpute import KNN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier



#Setting working directory
print(os.getcwd())
os.chdir("C:\\AI")


##Data import into environment
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")

#Adjusting columns
print(train_data.columns)
train_data['data']='train'
test_data['Loan_Status']='N'
test_data['data']='test'

#Concatenating data frame
merged_data=pd.concat([train_data,test_data])

##Missing value checking
missing_values=((merged_data.isnull().sum()/len(merged_data))*100)

##Imputing missing values
merged_data.dtypes
numeric_columns=merged_data.select_dtypes(['int64','float64'])
df_filled = pd.DataFrame(KNN(3).fit_transform(numeric_columns))
df_filled.columns=numeric_columns.columns
print(df_filled.columns)
df_filled['Credit_History']=np.where(df_filled['Credit_History']>0.5,1,0)
merged_data[df_filled.columns]=df_filled[df_filled.columns]

#

#Checking missing_values in test and train data
missing_values_train=((merged_data[merged_data['data']=='train'].isnull().sum()/len(merged_data[merged_data['data']=='train']))*100)
missing_values_test=((merged_data[merged_data['data']=='test'].isnull().sum()/len(merged_data[merged_data['data']=='test']))*100)

categorical_missing_columns=['Self_Employed','Dependents','Gender','Married']



merged_data_v1=merged_data.copy()

merged_data['Gender']=merged_data['Gender'].map({'Male':0,'Female':1})
merged_data['Married']=merged_data['Married'].map({'No':0,'Yes':1})
merged_data['Self_Employed']=merged_data['Self_Employed'].map({'No':0,'Yes':1})
merged_data['Dependents']=merged_data['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})
impute_data=merged_data[['Gender','Married','Self_Employed','Dependents']]
df_filled = pd.DataFrame(KNN(4).fit_transform(impute_data))
df_filled.columns=impute_data.columns
merged_data[df_filled.columns]=df_filled[df_filled.columns]


df_filled['Gender']=np.where(df_filled['Gender']>0.5,1,0)
df_filled['Married']=np.where(df_filled['Married']>0.5,1,0)
df_filled['Self_Employed']=np.where(df_filled['Self_Employed']>0.5,1,0)
df_filled['Dependents']=df_filled['Dependents'].apply(np.round)


df_filled['Gender']=df_filled['Gender'].map({0:'Male',1:'Female'})
df_filled['Married']=df_filled['Married'].map({0:'No',1:'Yes'})
df_filled['Self_Employed']=df_filled['Self_Employed'].map({0:'No',1:'Yes'})
#df_filled['Dependents']=df_filled['Dependents'].map({0:'0',1:'1',2:'2',3:'3+'})
merged_data[df_filled.columns]=df_filled[df_filled.columns]



merged_data_v2=merged_data.copy()

train_data_v1=merged_data[merged_data['data']=='train']
test_data_v1=merged_data[merged_data['data']=='test']

print(train_data_v1.columns)

###Data visualization
x=train_data_v1['Loan_Status']
y=train_data_v1['Loan_Status'].count()
pd.crosstab(train_data_v1['Loan_Status'],train_data_v1['Loan_Status'].count())


sns.boxplot(merged_data['ApplicantIncome'])

sns.boxplot(merged_data['CoapplicantIncome'])


sns.boxplot(merged_data['LoanAmount'])

pd.crosstab(train_data_v1['Gender'],train_data_v1['Loan_Status'].count())
pd.crosstab(train_data_v1['Married'],train_data_v1['Loan_Status'].count())




pd.crosstab(merged_data['Loan_Status'],merged_data['ApplicantIncome'].mean())
pd.crosstab(merged_data['Loan_Status'],merged_data['ApplicantIncome'].median())
pd.crosstab(merged_data['Loan_Status'],merged_data['CoapplicantIncome'].mean())
pd.crosstab(merged_data['Loan_Status'],merged_data['CoapplicantIncome'].median())


pd.crosstab(merged_data['Gender'],merged_data['ApplicantIncome'].mean())
pd.crosstab(merged_data['Gender'],merged_data['ApplicantIncome'].median())
pd.crosstab(merged_data['Gender'],merged_data['CoapplicantIncome'].mean())
pd.crosstab(merged_data['Gender'],merged_data['CoapplicantIncome'].median())


pd.crosstab(merged_data['Married'],merged_data['ApplicantIncome'].mean())
pd.crosstab(merged_data['Married'],merged_data['ApplicantIncome'].median())
pd.crosstab(merged_data['Married'],merged_data['CoapplicantIncome'].mean())
pd.crosstab(merged_data['Married'],merged_data['CoapplicantIncome'].median())


#Feature Engineering
#X=merged_data.groupby(['Dependents'])['ApplicantIncome'].mean().reset_index().rename(columns={'ApplicantIncome':'avg_appIncome_dependents'})
#merged_data=pd.merge(left=merged_data,right=X,how='inner',on='Dependents')



#X=merged_data.groupby(['Property_Area'])['ApplicantIncome'].mean().reset_index().rename(columns={'ApplicantIncome':'avg_appIncome_propertyarea'})
#merged_data=pd.merge(left=merged_data,right=X,how='inner',on='Property_Area')


#X=merged_data.groupby(['Property_Area'])['CoapplicantIncome'].mean().reset_index().rename(columns={'CoapplicantIncome':'coappIncome_propertyarea'})
#merged_data=pd.merge(left=merged_data,right=X,how='inner',on='Property_Area')


#X=merged_data.groupby(['Education'])['ApplicantIncome'].mean().reset_index().rename(columns={'ApplicantIncome':'avg_appIncome_Edu'})
#merged_data=pd.merge(left=merged_data,right=X,how='inner',on='Education')

X=merged_data.groupby(['Credit_History'])['ApplicantIncome'].mean().reset_index().rename(columns={'ApplicantIncome':'avg_appIncome_credit_hist'})
merged_data=pd.merge(left=merged_data,right=X,how='inner',on='Credit_History')


X=merged_data.groupby(['Credit_History'])['CoapplicantIncome'].mean().reset_index().rename(columns={'CoapplicantIncome':'avg_coappIncome_credit_hist'})
merged_data=pd.merge(left=merged_data,right=X,how='inner',on='Credit_History')


#X=merged_data.groupby(['Gender','Education','Self_Employed'])['ApplicantIncome'].mean().reset_index().rename(columns={'ApplicantIncome':'avg_appIncome_Gen_Edu_emp'})
#merged_data=pd.merge(left=merged_data,right=X,how='inner',on=['Gender','Education','Self_Employed'])


merged_data['tot_income']=merged_data['CoapplicantIncome']+merged_data['ApplicantIncome']
merged_data['Loan_per_month']=((merged_data['LoanAmount']/merged_data['Loan_Amount_Term'])*1000)
merged_data['Income_out_permonth']=(merged_data['Loan_per_month']/merged_data['tot_income'])
#merged_data['loan_tot_income_ratio']=(merged_data['LoanAmount']/merged_data['tot_income'])


merged_data['total_number_persons']=9999
for i in range(0,len(merged_data)-1):
    if ((merged_data['ApplicantIncome'][i]>0) & (merged_data['CoapplicantIncome'][i]>0)):
        merged_data['total_number_persons'][i]=2+merged_data['Dependents'][i]
    
    elif((merged_data['ApplicantIncome'][i]>0) & (merged_data['CoapplicantIncome'][i]<=0)):
        merged_data['total_number_persons'][i]=1+merged_data['Dependents'][i]

merged_data['percapita_income_per_person']=((merged_data['tot_income']/merged_data['total_number_persons']))
    
    







'''['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status',
       'data', 'avg_appIncome_dependents', 'avg_appIncome_propertyarea',
       'avg_appIncome_Edu', 'avg_appIncome_Gen_Edu_emp']'''
##Converting categorical into numerical variables

final_data=pd.get_dummies(data=merged_data,columns=['Gender','Married','Dependents','Education','Self_Employed',
                                                    'Property_Area','Credit_History'])
##Modelling
train_set=final_data[final_data['data']=='train']
y=train_set['Loan_Status']
X=train_set.drop(columns=['Loan_ID','Loan_Status','data'],axis=1)
y=np.where(y=='Y',1,0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=y)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
predictions=gb.predict(X_test)
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))


from sklearn.model_selection import GridSearchCV

parameters = {
    "learning_rate": [0.025, 0.1],
    "min_samples_leaf":[0.1, 0.5],
    "max_depth":[3,5,8],
    "criterion": ["friedman_mse","mae"],
    "subsample":[ 0.8,0.9, 1.0],
    "n_estimators":[50,100,150]
    }
clf = GridSearchCV(gb, parameters, cv=10,verbose=2)

clf.fit(X_train, y_train)
print(clf.best_params_)


gb = GradientBoostingClassifier(criterion='friedman_mse', learning_rate=0.025, max_depth=3, min_samples_leaf=0.1, n_estimators= 50, subsample=0.8)
gb.fit(X_train, y_train)
predictions=gb.predict(X_test)
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))

predictions=gb.predict(X_train)
print(classification_report(y_train,predictions))
print(accuracy_score(y_train,predictions))

feature_importance=pd.DataFrame(gb.feature_importances_,X_train.columns).reset_index().rename(columns={'index':'Column_name',0:'Feature_importance'})
feature_importance=feature_importance.sort_values(['Feature_importance'],ascending=False).iloc[0:10,]
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10
plt.bar(feature_importance['Column_name'],feature_importance['Feature_importance'])





validation_set=final_data[final_data['data']=='test']

validation_set_1=final_data[final_data['data']=='test']

validation_set=validation_set[X_train.columns]
predictions=pd.DataFrame(gb.predict(validation_set))
predictions=np.where(predictions.iloc[:,0]==0,'N','Y')



import lightgbm as lgb

train_data=lgb.Dataset(X_train,label=y_train)

param = {'num_leaves':150, 'objective':'binary','max_depth':7,'learning_rate':.05,'max_bin':200}
param['metric'] = ['auc', 'binary_logloss']

num_round=50

lgbm=lgb.train(param,train_data,num_round)
predictions=lgbm.predict(X_test)
for i in range(0,len(predictions)-1):
    if predictions[i]>=.5:       # setting threshold to .5
       predictions[i]=1
    else:  
       predictions[i]=0
       
predictions=predictions.astype(int)       
print(classification_report(predictions,y_test))
print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))


