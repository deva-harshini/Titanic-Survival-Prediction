🚢 Titanic Survival Prediction
📌 Problem Statement
The Titanic dataset is one of the most famous beginner-friendly datasets in data science.
The challenge is to build a binary classification model that predicts whether a passenger survived or not, based on features such as age, sex, class, family size, and fare.

📊 Dataset
Train dataset: train.csv (labeled with survival outcome)
Test dataset: test.csv (unlabeled, for submission)
Source: Kaggle Titanic: Machine Learning from Disaster
⚙️ Methods & Workflow
Data Preprocessing

Handled missing values (Age, Fare, Embarked)
Dropped Cabin due to high missingness
Feature Engineering

Created FamilySize, IsAlone, Title, AgeBin, FareBin
Encoded categorical variables (Sex, Embarked, Title)
Modeling

Logistic Regression
Decision Tree
Random Forest ✅ (best)
Gradient Boosting
Support Vector Machine
Evaluation

Metrics: Accuracy, Precision, Recall, F1-score
Best Model: Random Forest (~82% accuracy)
📈 Visualizations
Some insights from the data:

Women had a survival rate of ~74%
First-class passengers had much higher survival chances than third-class
Passengers traveling alone had lower survival rates
(See TitanicVisualization.py for plots)

🛠️ Usage Guide
1️⃣ Clone the repository
git clone https://github.com/deva-harshini/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
