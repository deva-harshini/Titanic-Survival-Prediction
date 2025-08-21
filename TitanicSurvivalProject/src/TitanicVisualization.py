import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("train.csv")

# Missing values
plt.figure(figsize=(8,5))
sns.heatmap(train_df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# Survival distribution
sns.countplot(x="Survived", data=train_df, palette="viridis")
plt.title("Survival Count")
plt.show()

# Survival by gender
sns.barplot(x="Sex", y="Survived", data=train_df, palette="viridis")
plt.title("Survival by gender")
plt.show()

# Survival by Class
sns.barplot(x="Pclass", y="Survived", data=train_df, palette="viridis")
plt.title("Survival by Passenger Class")
plt.show()

# Age distribution
sns.histplot(train_df["Age"].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.show()
