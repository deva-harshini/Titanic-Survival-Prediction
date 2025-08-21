import pandas as pd
import joblib

# Load model + features
model = joblib.load("titanic_best_model.pkl")
features = joblib.load("titanic_features.pkl")

# Load test data
test_df = pd.read_csv("test.csv")

# Preprocessing
test_df["Age"].fillna(test_df["Age"].median(), inplace=True)
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
test_df.drop("Cabin", axis=1, inplace=True)

test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
test_df["IsAlone"] = (test_df["FamilySize"] == 1).astype(int)

test_df["Title"] = test_df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
test_df["Title"] = test_df["Title"].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Other')
test_df["Title"] = test_df["Title"].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})

test_df["AgeBin"] = pd.cut(test_df["Age"], bins=[0,12,20,40,60,80],
                           labels=["Child","Teen","Adult","MiddleAge","Senior"])
test_df["FareBin"] = pd.qcut(test_df["Fare"], 4, labels=["Low","Mid","High","VeryHigh"])

test_df["Sex"] = test_df["Sex"].map({"male":1,"female":0})
test_df["Embarked"] = test_df["Embarked"].map({"S":0,"C":1,"Q":2})
test_df["Title"] = test_df["Title"].map({"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Other":4})
test_df["AgeBin"] = test_df["AgeBin"].map({"Child":0,"Teen":1,"Adult":2,"MiddleAge":3,"Senior":4})
test_df["FareBin"] = test_df["FareBin"].map({"Low":0,"Mid":1,"High":2,"VeryHigh":3})

# Prediction
X_test_final = test_df[features]
predictions = model.predict(X_test_final)

submission = pd.DataFrame({"PassengerId":test_df["PassengerId"],"Survived":predictions})
submission.to_csv("titanic_submission.csv", index=False)

print("✅ Kaggle submission file 'titanic_submission.csv' created!")
