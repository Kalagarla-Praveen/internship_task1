# Titanic Dataset - Data Cleaning & Preprocessing

#Objective
The goal of this project is to perform **data cleaning and preprocessing** on the Titanic dataset in preparation for machine learning tasks. This includes handling missing values, encoding categorical features, feature scaling, and detecting/removing outliers.

---

##  Tools & Libraries Used
- **Python**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib / Seaborn**: Data visualization
- **Scikit-learn**: Preprocessing and scaling

---

## 📁 Dataset
The dataset used is the **Titanic Dataset**, which contains information about passengers aboard the Titanic. It includes both categorical and numerical features.

---

## 🧪 Steps Performed

### 1. Data Import & Exploration
- Loaded the CSV file using `pandas.read_csv()`.
- Displayed the first few rows of the dataset using `df.head()`.
- Used `df.info()` and `df.isnull().sum()` to check for data types and missing values.

### 2. Handling Missing Values
- **Age** column had missing values – filled using **median imputation**.
- **Embarked** column had missing values – filled using the **mode**.
- **Cabin** column was dropped due to excessive missing values.
- Ensured no missing values remained in the **target column** (`Survived`).

### 3. Encoding Categorical Features
- Converted **Sex** and **Embarked** columns into numerical format using **one-hot encoding** (`pd.get_dummies()`), dropping the first category to avoid dummy variable trap.

### 4. Feature Scaling (Normalization)
- Standardized numerical features (**Age**, **Fare**, **SibSp**, and **Parch**) using **`StandardScaler`** from Scikit-learn to bring them onto the same scale.

### 5. Outlier Detection & Removal
- Used **boxplots** to visualize potential outliers in numerical features.
- Focused on **Fare**: Removed extreme outliers using the **Interquartile Range (IQR)** method to improve data quality.

---

## 📈 Final Outcome
The dataset is now:
- Cleaned with no missing values.
- Ready with numerical encoding for categorical variables.
- Standardized for machine learning algorithms.
- Outliers handled appropriately to ensure model robustness.

---

## 📌 Next Steps
This cleaned dataset is now ready for:
- Exploratory Data Analysis (EDA)
- Feature engineering
- Training ML models like Logistic Regression, Decision Trees, etc.

---


