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

# Hypothesis until now:
## The "age" of the customer doesn't look to influence so much in the problem because there are no atypical behavior in the univariate analysis and the medians are almost in the same point (in the candle plot).

## The "monthUse" looks to be a strong feature because it looks that the highest rates of churn are concentrated in customers that are not using to much the service.
## Other thing we noticed is that there are some excepctions that use a lot the service and cancel, it's interesting to take a look at those cases to try to find a pattern to this subgroup.

## The "tier" shows that the "Premium" group has the biggest churn rate. Event if this rate is not very expressive (in comparison to other categories) maybe we can find somethings in this "tier" that is not pleasing the customers

## "Satisfaction" show us the churn has a big correlation to the score the customer gave on the satisfcation pool. This reinforces the idea that come customers cancel because they are unhappy.

## "Time" doesn't looks to be a feature of big importance even showing a small tendence of longer contracts be canceled.

## "MonthPayment" doesn't looks to be a feature of big importance even showing a small tendence of who pay less.

### SPLIT INTO TRAIN AND TEST ###

def split_dataset(data, target_column, test_size, random_state = 42):
    '''
    Splits the dataset into train and test.

    Parameters:
    - data (DataFrame): The full dataframe.
    - target_column (str): The name of the target column.
    - test_size (float): The proportion used to test.
    - random_state (int): Seed to genereate aleatory numbers (The default value is 42).

    Return:
    - X_train (DataFrame): Train dataset.
    - X_test (DataFrame): Test dataset.
    - y_train (Series): Target values for train dataset.
    - y_test (Series): Target values for test dataset.
    '''

    # Split the features from the target into "inputs" and "outputs"
    x = data.drop(target_column,axis = 1)

    # Output data
    y = data[target_column]

    # Split in train and test
    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    
    return X_train, X_test, y_train, y_test

value_test_size = 0.3
X_train, X_test, y_train, y_test = split_dataset(df, 'Churn', test_size=value_test_size)

print(f"The X_train size is: {X_train.shape}")
print(f"The X_test size is: {X_test.shape}")
print(f"The y_train size is: {y_train.shape}")
print(f"The y_test size is: {y_test.shape}")

### PREPROCESSING DATA ###

# Filtering cathegorical features
categorical_cols = df.select_dtypes(include = ['object']).columns
print(categorical_cols)

## One-Hot Encoding ##
# Apply the One-Hot encoding separately on train and test
encoder = OneHotEncoder(sparse_output=False)

# Trainning the encoder with the train data and apply in both
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_cols]))
X_test_encoded = pd.DataFrame(encoder.fit_transform(X_test[categorical_cols]))

# Rename the columns
X_train_encoded.columns = encoder.get_feature_names_out(categorical_cols)
X_test_encoded.columns = encoder.get_feature_names_out(categorical_cols)

# Remove the original columns
X_train_preprocessed = X_train.drop(categorical_cols, axis = 1).reset_index(drop = True)
X_train_preprocessed = pd.concat([X_train_preprocessed, X_train_encoded], axis = 1)
print('X_train_preprocessed')
print(X_train_preprocessed.head())

X_test_preprocessed = X_test.drop(categorical_cols, axis = 1).reset_index(drop = True)
X_test_preprocessed = pd.concat([X_test_preprocessed, X_test_encoded], axis = 1)
print('X_test_preprocessed')
print(X_test_preprocessed.head())

## StandardScaler ##

# Selecting the numeric features
numeric_cols = X_train_preprocessed.select_dtypes(include=['int64', 'float64']).columns

# Creating the StandardScaler
scaler = StandardScaler()

# Apply the StandardScaler
X_train_preprocessed[numeric_cols] = scaler.fit_transform(X_train_preprocessed[numeric_cols])
X_test_preprocessed[numeric_cols] = scaler.transform(X_test_preprocessed[numeric_cols])

print('X_train_preprocessed')
print(X_train_preprocessed.head())

print('X_test_preprocessed')
print(X_test_preprocessed.head())