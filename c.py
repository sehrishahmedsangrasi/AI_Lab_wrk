import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("test.csv")

# Select relevant columns for x (replace 'zero' with actual column names if needed)
x = df[["Passengerid", "Age", "Fare", "Sex", "sibsp", "Parch", "Pclass", "Embarked"]]

# Target variable (ensure correct column name)
y = df["2urvived"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalize the data using the same MinMaxScaler for both training and test sets
scaler = MinMaxScaler().fit(X_train)

X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)

# Convert normalized data back to DataFrame
X_train_norm_df = pd.DataFrame(X_train_norm, columns=X_train.columns)
X_test_norm_df = pd.DataFrame(X_test_norm, columns=X_test.columns)

# Save the normalized training and test sets to CSV files
X_train_norm_df.to_csv('test_train.csv', header=True, index=False)
X_test_norm_df.to_csv('test_test.csv', header=True, index=False)
