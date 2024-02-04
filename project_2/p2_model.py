import joblib # Used to save and read the .pkl files 
import sklearn # Framework to create the machine learning model
import pandas as pd # Used to work with dataframes
import numpy as np # Used to work with arrays
import matplotlib.pyplot as plt # Used to create simple plots
import seaborn as sns # An "upgraded" version of the matplotlib helping to make plots of regressions
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

### READING DATA ###
df = pd.read_csv('project_2/dataset.csv')

# Check the size of the dataset (rows , columns)
print(df.shape)

# Check the type of each column
print(df.info())

# Show the first lines of the dataset
print(df.head())

### EDA (EXPLORATORY DATA ANALYSIS) ###
def eda(data):
    for column in data.columns:
        
        # If the column is numeric
        if data[column].dtype in ['int64', 'float64']:

            #Histogram and Boxplot
            fig, axes = plt.subplots(1,2)
            sns.histplot(data[column], kde = True, ax = axes[0])
            sns.boxplot(x = 'Churn', y = column, data = data, ax = axes[1])
            axes[0].set_title(f'Distribution of {column}')
            axes[1].set_title(f'{column} vs Churn')
            plt.tight_layout()
            plt.show()
            
        # If the column is categorical    
        else:

            # Frequency count related to Churn
            fig, axes = plt.subplots(1,2)
            sns.countplot(x = column, data = data, ax=axes[0])
            sns.countplot(x = column, hue = 'Churn', data = data, ax = axes[1])
            axes[0].set_title(f'Distribution of {column}')
            axes[1].set_title(f'{column} vs Churn')
            plt.tight_layout()
            plt.show()

eda(df)            