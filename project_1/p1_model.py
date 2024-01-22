# Project 1 - Construction and Deployment of Machine Learning Models
## Objective of this script: Creation of the machine learning model

# Imports
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Products data
dsa_data = {
    'Package_Weight': [212, 215, 890, 700, 230, 240, 730, 780, 218, 750, 202, 680],
    'Package_Type': ['Cardboard', 'Cardboard', 'Bubble_Wrap', 'Bubble_Wrap', 'Cardboard', 'Cardboard', 'Bubble_Wrap', 'Bubble_Wrap', 'Cardboard', 'Bubble_Wrap', 'Cardboard', 'Bubble_Wrap'],
    'Product_Type': ['Smartphone', 'Tablet', 'Tablet', 'Tablet', 'Smartphone', 'Smartphone', 'Tablet', 'Smartphone', 'Smartphone', 'Tablet', 'Smartphone', 'Tablet'] 
}

# Converting the dictionary into dataframe
df = pd.DataFrame(dsa_data)

# Split X (input) and Y (output)
X = df[['Package_Weight', 'Package_Type']]
y = df['Product_Type']

# Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Creation and fit the transformations in the training data

# Fit of the categorical feature "Package_Type"
le_package_type = LabelEncoder() #Assing a number to each category
le_package_type.fit(X_train['Package_Type']) # Using the train data, I train my "encoder"

# Fit of the categorical feature "Product_Type"
le_product_type = LabelEncoder()
le_product_type.fit(y_train)

# Apply the transformation in the train and test data
X_train['Package_Type'] = le_package_type.transform(X_train['Package_Type'])
X_train['Package_Type'] = le_package_type.transform(X_test['Package_Type'])

y_train = le_product_type.transform(y_train)
y_test = le_product_type.transform(y_test)

