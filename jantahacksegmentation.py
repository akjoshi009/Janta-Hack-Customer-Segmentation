"""
Created on Sat Aug  1 19:16:24 2020

@author: Sachin Anbhule
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier


link=r"D:\Sachin\Data Science Challenges\Janta Hack Customer Segmentation"
train=pd.read_csv(link+"\\Train.csv")
test=pd.read_csv(link+"\\Test.csv")

train.columns

train.isnull().sum()
train["Profession"].value_counts()

train["Ever_Married"]=train["Ever_Married"].fillna("Yes")
train["Graduated"]=train["Graduated"].fillna("Yes")
train["Profession"]=train["Profession"].fillna("No_idea")
train["Work_Experience"]=train["Work_Experience"].fillna(train["Work_Experience"].median())
train["Family_Size"]=train["Family_Size"].fillna(train["Family_Size"].median())


test["Ever_Married"]=test["Ever_Married"].fillna("No")
test["Graduated"]=test["Graduated"].fillna("No")
test["Profession"]=test["Profession"].fillna("No_idea")
test["Work_Experience"]=test["Work_Experience"].fillna(test["Work_Experience"].median())
test["Family_Size"]=test["Family_Size"].fillna(test["Family_Size"].median())


train["Var_1"]=train["Var_1"].astype(str)
test["Var_1"]=test["Var_1"].astype(str)

train["Var_1"]=train["Var_1"].replace('nan','Cat_6')
test["Var_1"]=test["Var_1"].replace('nan','Cat_6')


train["Gender"]=train["Gender"].map({'Male':0, 'Female': 1})
train["Ever_Married"]=train["Ever_Married"].map({'No':0, 'Yes': 1})
train["Graduated"]=train["Graduated"].map({'No':0, 'Yes': 1})
train["Spending_Score"]=train["Spending_Score"].map({'Low':0, 'Average': 1,'High':2})
train["Var_1"]=train["Var_1"].map({'Cat_6':0, 'Cat_4': 1,'Cat_3':2,'Cat_2':3,'Cat_7':4,'Cat_1':5,'Cat_5':6})


test["Gender"]=test["Gender"].map({'Male':0, 'Female': 1})
test["Ever_Married"]=test["Ever_Married"].map({'No':0, 'Yes': 1})
test["Graduated"]=test["Graduated"].map({'No':0, 'Yes': 1})
test["Spending_Score"]=test["Spending_Score"].map({'Low':0, 'Average': 1,'High':2})
test["Var_1"]=test["Var_1"].map({'Cat_6':0, 'Cat_4': 1,'Cat_3':2,'Cat_2':3,'Cat_7':4,'Cat_1':5,'Cat_5':6})


# =============================================================================
# le = preprocessing.LabelEncoder()
# le.fit(train["Profession"])
# train["Profession"]=le.transform(train["Profession"])
# test["Profession"]=le.transform(test["Profession"])
# =============================================================================


def getrealexpe(row):
    if(row["Age"] - row["Work_Experience"]<18):
        return row["Age"]-18
    else:
        return row["Work_Experience"]
    
    

train["realexpe"]=train.apply(getrealexpe,axis=1)
test["realexpe"]=test.apply(getrealexpe,axis=1)



#train["Family_Size"].value_counts()

train["Age"]=np.log(train["Age"])
test["Age"]=np.log(test["Age"])

#train["Family_Size"]=np.where(train["Family_Size"]>4.0,5.0,train["Family_Size"])
#test["Family_Size"]=np.where(test["Family_Size"]>4.0,5.0,test["Family_Size"])
test = pd.merge(test, train[['ID', 'Segmentation']], on='ID', how='left')
#test.isnull().sum()
train=pd.get_dummies(train,prefix_sep="_",columns=['Profession'])
test=pd.get_dummies(test,prefix_sep="_",columns=['Profession'])

train.columns

testx=test[test["Segmentation"].isnull()]
scaler = StandardScaler()
colx=['ID','Gender', 'Ever_Married', 'Age', 'Graduated', 'Work_Experience',
       'Spending_Score', 'Family_Size', 'Var_1', 'realexpe',
       'Profession_Artist', 'Profession_Doctor', 'Profession_Engineer',
       'Profession_Entertainment', 'Profession_Executive',
       'Profession_Healthcare', 'Profession_Homemaker', 'Profession_Lawyer',
       'Profession_Marketing', 'Profession_No_idea'
       ]

scaler.fit(train[colx])
ttrain=scaler.transform(train[colx])
ttest=scaler.transform(testx[colx])

params = {
    'learning_rate': 0.12, 
    'max_depth': 3, 
    'min_data_in_leaf': 19, 
    'n_estimators': 376, 
    'reg_alpha': 1.05, 
    'reg_lambda': 2.53, 
    'objective': 'multiclass', 
    'boosting_type': 'gbdt', 
    'subsample': 0.7, 
    'random_state': 42, 
    'colsample_bytree': 0.7
    }


X_train, X_test, y_train, y_test = train_test_split(
    ttrain,train['Segmentation'], test_size=0.2, random_state=42)
    
clf=LGBMClassifier(**params)
clf.fit(X_train,y_train)
p=clf.predict(X_test)
accuracy_score(y_test,p)
confusion_matrix(y_test,p)

plt.barh(colx,clf.feature_importances_)

clf.fit(ttrain,train["Segmentation"])
pred=clf.predict(ttest)
testx["Segmentation"]=pred

test = pd.merge(test, testx[['ID', 'Segmentation']], on='ID', how='left')
test["Segmentation"]=np.where(test["Segmentation_x"].isnull(),test["Segmentation_y"],test["Segmentation_x"])
test[["ID","Segmentation"]].to_csv(link+"\output.csv",index=False)
