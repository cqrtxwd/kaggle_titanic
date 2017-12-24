import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

titanic_df = pd.read_csv("E:/kaggle/Titanic/train.csv")
test_df    = pd.read_csv("E:/kaggle/Titanic/test.csv")

#show basic info

print(titanic_df.head())
titanic_df.info()

print("--------------------------------------------------")
#drop useless cols
titanic_df=titanic_df.drop(["PassengerId","Name","Ticket"],axis=1) 
print(titanic_df.head())

print("---------------------Embarked-----------------------------")
#Embarked
#fill nan with most likely data
print(titanic_df["Embarked"].describe())
titanic_df["Embarked"].fillna("S")

#compute each Embark survived rate
embarked_statics=titanic_df[["Survived","Embarked"]].groupby(["Embarked"]).mean()
embarked_one_hot=pd.get_dummies(titanic_df["Embarked"])
titanic_df=titanic_df.join(embarked_one_hot)
titanic_df.drop(["Embarked"],axis=1,inplace=True)

print(titanic_df.head())
print("-----------------------------fare---------------------------------------")

#fare
#convert double to int

most_fare=titanic_df["Fare"].median()
#titanic_df["Fare"].fillna(most_fare);

titanic_df["Fare"]=titanic_df["Fare"].astype(int)

#analys contributes of fare to survived
print("fare mean=\n" ,titanic_df[["Fare","Survived"]].groupby(["Survived"]).mean(),"\n")
#print("max fare=\n" ,titanic_df[["Fare","Survived"]].groupby(["Survived"]).max())

fare_survived=titanic_df["Fare"][titanic_df["Survived"]==1]
fare_not_survived=titanic_df["Fare"][titanic_df["Survived"]==0]

print(fare_survived.mean())
print(fare_not_survived.mean())
print("----------------------------------------Age--------------------------------------------------------")

#age

#get basic info of ages
age_mean=titanic_df["Age"].mean()
age_std=titanic_df["Age"].std()
age_nan_count=titanic_df["Age"].isnull().sum()

print(age_mean)
print(age_std)
print(age_nan_count)
print(titanic_df["Age"].count())

#fill the missing age data with generated fake age
rand=np.random.randint(age_mean-age_std,age_mean+age_std,size=age_nan_count)

titanic_df["Age"][titanic_df["Age"].isnull()]=rand;
# or use the code below instead
#titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand;

#convert Age to int
titanic_df["Age"]=titanic_df["Age"].astype(int)

print("------------------------------Cabin------------------------------------------")
#Cabin
#there is too much nan value,so drop the whole line

#titanic_df=titanic_df.drop(["Cabin"],axis=1)
titanic_df.drop(["Cabin"],axis=1,inplace=True) #titanic_df=titanic_df.drop(["Cabin"],axis=1)

print("------------------------------Parch & Sibsp------------------------------------------")
# Parch & Sibsp
#
titanic_df["Family"]=titanic_df["SibSp"]+titanic_df["Parch"];

#generate family feature data
#titanic_df["Family"][titanic_df["Family"]>=1]=1;
#titanic_df["Family"][titanic_df["Family"]==0]=0;
#
titanic_df["Family"].loc[titanic_df["Family"]>0]=1;
titanic_df["Family"].loc[titanic_df["Family"]==0]=0;

#drop Parch & Sibsp cols
titanic_df=titanic_df.drop(["Parch","SibSp"],axis=1)

print("--------------------------------Sex-------------------------------------------------------")
def get_new_sex(person):
	age,sex=person;
	return "Child" if age<=16 else sex; 

titanic_df['Sex'] = titanic_df[['Age','Sex']].apply(get_new_sex,axis=1)
sex_one_hot=pd.get_dummies(titanic_df["Sex"]);
sex_one_hot.columns=["Child","Female","Male"]
titanic_df=titanic_df.join(sex_one_hot);
titanic_df.drop(["Sex"],axis=1,inplace=True)

print("--------------------------------Pclass-------------------------------------------------------")
#Pclass
Pclass_one_hot=pd.get_dummies(titanic_df["Pclass"])
Pclass_one_hot.columns=["Pclass1","Pclass2","Pclass3"];
titanic_df=titanic_df.join(Pclass_one_hot)
titanic_df=titanic_df.drop(["Pclass"],axis=1)

print("-------------------------------- data after process -------------------------------------------------------")
print(titanic_df.head(30))
titanic_df.info()
print("------------------------- start train and test classifier -------------------------------")

X_train=titanic_df.drop(["Survived"],axis=1)[:720]
Y_train=titanic_df["Survived"][:720];
X_valid=titanic_df.drop(["Survived"],axis=1)[720:]
Y_valid=titanic_df["Survived"][720:];

#logistic regression
G=LogisticRegression();
G.fit(X_train,Y_train);
score_train=G.score(X_train,Y_train)
score_valid=G.score(X_valid,Y_valid)
print("LogisticRegression:")
print("train score= \n",score_train)
print("validation score= \n",score_valid,"\n")

#SVM
G=SVC();
G.fit(X_train,Y_train);
score_train=G.score(X_train,Y_train)
score_valid=G.score(X_valid,Y_valid)
print("SVM:")
print("train score= \n",score_train)
print("validation score= \n",score_valid,"\n")

#random forest
G=RandomForestClassifier();
G.fit(X_train,Y_train);
score_train=G.score(X_train,Y_train)
score_valid=G.score(X_valid,Y_valid)
print("RandomForestClassifier:")
print("train score= \n",score_train)
print("validation score= \n",score_valid,"\n")

#KNN
G=KNeighborsClassifier();
G.fit(X_train,Y_train);
score_train=G.score(X_train,Y_train)
score_valid=G.score(X_valid,Y_valid)
print("KNeighborsClassifier:")
print("train score= \n",score_train)
print("validation score= \n",score_valid,"\n")


#gaussian naive bayes
G=GaussianNB();
G.fit(X_train,Y_train);
score_train=G.score(X_train,Y_train)
score_valid=G.score(X_valid,Y_valid)
print("GaussianNB:")
print("train score= \n",score_train)
print("validation score= \n",score_valid,"\n")


print("---------------------- get the coeffs ----------------------------------------------------")

G=LogisticRegression();
G.fit(X_train,Y_train);
score_train=G.score(X_train,Y_train)
score_valid=G.score(X_valid,Y_valid)
print("LogisticRegression:")
print("train score= \n",score_train)
print("validation score= \n",score_valid,"\n")

coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns=["features"];
coeff_df["weights"]=G.coef_[0];

print(coeff_df)
