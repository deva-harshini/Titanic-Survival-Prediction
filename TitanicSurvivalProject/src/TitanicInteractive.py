import pandas as pd
import joblib

# Load trained model & features
model = joblib.load("titanic_best_model.pkl")
features = joblib.load("titanic_features.pkl")

print("\n🚢 Titanic Survival Prediction System 🚢")
print("Enter passenger details:")

# Collect inputs
pclass = int(input("Passenger Class (1, 2, 3): "))
sex_input = input("Sex (male/female): ").lower()
sex = 1 if sex_input == "male" else 0
age = float(input("Age: "))
sibsp = int(input("Siblings/Spouses aboard: "))
parch = int(input("Parents/Children aboard: "))
fare = float(input("Fare: "))
embarked_input = input("Embarked (S/C/Q): ").upper()
embarked = {"S":0,"C":1,"Q":2}.get(embarked_input,0)

# Feature Engineering
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

if age < 12:
    title = 3
elif sex == 0 and age >= 18:
    title = 2
elif sex == 0:
    title = 1
else:
    title = 0

if age <= 12: age_bin = 0
elif age <= 20: age_bin = 1
elif age <= 40: age_bin = 2
elif age <= 60: age_bin = 3
else: age_bin = 4

if fare <= 8: fare_bin = 0
elif fare <= 15: fare_bin = 1
elif fare <= 30: fare_bin = 2
else: fare_bin = 3

# Final DataFrame
new_passenger = pd.DataFrame([{
    "Pclass":pclass,"Sex":sex,"Age":age,"SibSp":sibsp,"Parch":parch,
    "Fare":fare,"Embarked":embarked,"FamilySize":family_size,
    "IsAlone":is_alone,"Title":title,"AgeBin":age_bin,"FareBin":fare_bin
}])[features]

# Prediction
prediction = model.predict(new_passenger)[0]
print("\n🎯 Prediction:", "Survived ✅" if prediction==1 else "Did Not Survive ❌")
