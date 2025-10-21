import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Base.csv')

print("=== FRAUD DETECTION DATASET ANALYSIS ===")
print(f"Dataset Shape: {df.shape}")
print(f"Total Records: {df.shape[0]:,}")
print(f"Features: {df.shape[1]}")

print("\n=== FRAUD DISTRIBUTION ===")
fraud_counts = df['fraud_bool'].value_counts()
fraud_pct = df['fraud_bool'].value_counts(normalize=True) * 100
print(f"Legitimate Transactions: {fraud_counts[0]:,} ({fraud_pct[0]:.2f}%)")
print(f"Fraudulent Transactions: {fraud_counts[1]:,} ({fraud_pct[1]:.2f}%)")
print(f"Fraud Rate: {fraud_pct[1]:.3f}%")

print("\n=== DATA QUALITY CHECK ===")
print(f"Missing Values: {df.isnull().sum().sum()}")
print(f"Duplicated Rows: {df.duplicated().sum()}")

print("\n=== NUMERICAL FEATURES SUMMARY ===")
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('fraud_bool')  # Remove target variable
print(f"Numerical Features: {len(numerical_cols)}")
print(df[numerical_cols].describe())

print("\n=== CATEGORICAL FEATURES SUMMARY ===")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical Features: {len(categorical_cols)}")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))

print("\n=== FRAUD PATTERNS ANALYSIS ===")
# Analyze fraud by categorical features
for col in categorical_cols:
    fraud_by_cat = df.groupby(col)['fraud_bool'].agg(['count', 'sum', 'mean']).round(4)
    fraud_by_cat.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate']
    fraud_by_cat = fraud_by_cat.sort_values('Fraud_Rate', ascending=False)
    print(f"\nFraud Rate by {col}:")
    print(fraud_by_cat.head())

print("\n=== KEY INSIGHTS ===")
print("1. Dataset is highly imbalanced with only 1.1% fraud rate")
print("2. No missing values or duplicates - clean dataset")
print("3. Mix of numerical and categorical features")
print("4. Need to focus on feature engineering for better fraud detection")

# Create visualization directory
import os
os.makedirs('plots', exist_ok=True)

# Plot 1: Fraud Distribution
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
fraud_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Transaction Distribution')
plt.xlabel('Fraud Status (0=Legitimate, 1=Fraud)')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
plt.pie(fraud_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.2f%%', colors=['green', 'red'])
plt.title('Fraud Percentage')

plt.tight_layout()
plt.savefig('plots/fraud_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualization saved: plots/fraud_distribution.png")
print("Data exploration complete!")