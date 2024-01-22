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
X_test['Package_Type'] = le_package_type.transform(X_test['Package_Type'])

y_train = le_product_type.transform(y_train)
y_test = le_product_type.transform(y_test)

# Creation of the model
dsa_model = DecisionTreeClassifier()

# Train the model
dsa_model.fit(X_train, y_train)

# Do the prediction with the model
y_pred = dsa_model.predict(X_test)

# Calculate accuracy: In 100 predictions, I scored 67 right
## Over 50% of accuracy = it's possible to be used
## Above 70% of accureacy = it can be improved
## Over 70% of accuracy = it's starting to become good

acc_dsa_mode = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: ", round(acc_dsa_mode, 2))

# Metrics
## Accuracy: From all the predictions, how much the model answered "right" correctly.
## Precision: From what the model predicted as "right", how many was correct.
## Recall: From what is really right, how much the modl said "right" and was correct.
## f1-score: It makes a harmonical mean of Precision and Recall, givin importance to low values and showing us if something is unbalanced.
print(f"\n Classification report: \n")
report = classification_report(y_test, y_pred)
print(report)

# Save the trained model
joblib.dump(dsa_model, 'project_1/models/logistic_model.pkl')

# Save the transformations
joblib.dump(le_package_type, 'project_1/models/transformation_package_type.pkl')
joblib.dump(le_product_type, 'project_1/models/transformation_product_type.pkl')