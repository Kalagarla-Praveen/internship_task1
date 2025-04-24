import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("./Titanic-Dataset.csv")

# Display first 5 rows
print(df.head())

# Basic info
print(df.info())

# Check for missing values
print(df.isnull().sum())
