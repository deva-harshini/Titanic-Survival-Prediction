# TitanicExploration.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
train_df = pd.read_csv("E:/TitanicSurvivalProject/train.csv")

# Handlke missing values
train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)
train_df.drop("Cabin", axis=1, inplace=True)

# Feature engineering
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df["IsAlone"] = (train_df["FamilySize"] == 1).astype(int)
train_df["Title"] = train_df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
train_df["Title"] = train_df["Title"].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Other')
train_df["Title"] = train_df["Title"].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})

train_df["AgeBin"] = pd.cut(train_df["Age"], bins=[0,12,20,40,60,80],
                             labels=["Child","Teen","Adult","MiddleAge","Senior"])
train_df["FareBin"] = pd.qcut(train_df["Fare"], 4, labels=["Low","Mid","High","VeryHigh"])

# Encoding
train_df["Sex"] = train_df["Sex"].map({"male":1,"female":0})
train_df["Embarked"] = train_df["Embarked"].map({"S":0,"C":1,"Q":2})
train_df["Title"] = train_df["Title"].map({"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Other":4})
train_df["AgeBin"] = train_df["AgeBin"].map({"Child":0,"Teen":1,"Adult":2,"MiddleAge":3,"Senior":4})
train_df["FareBin"] = train_df["FareBin"].map({"Low":0,"Mid":1,"High":2,"VeryHigh":3})

#Feature Selection
features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked",
            "FamilySize","IsAlone","Title","AgeBin","FareBin"]
X = train_df[features]
y = train_df["Survived"]

# Train/Split set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Mode; Training
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC()
}

results = {}
for name, model in models.items():
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test,preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Best model selection
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✅ Best model is: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save model + features
joblib.dump(best_model,"titanic_best_model.pkl")
joblib.dump(features,"titanic_features.pkl")
