# This script was created to check python and packages version

import sys # Used to have accesss to the Python system variables (like which system is running your code).
import joblib # Used to parallelize tasks, in our case it is used specially to import machine learning models.
import pandas # Used to work with dataframes.
import sklearn # Contains everything we will need to develop the Machine Learning Model.
import flask # This tool is used to develop the application where the model will be plugged.

packages = [joblib, pandas, sklearn, flask]

# Get the string of the python version
version_string = sys.version
version_number = version_string.split()[0]

print("Python Version:", version_number)
for package in packages:
    print(f"Version of {package.__name__}:", package.__version__)

