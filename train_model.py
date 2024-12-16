# train_model.py
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
loan_dataset = pd.read_csv('dataset.csv')  # Update the path to your dataset

# Check for missing values
missing_values_count = loan_dataset.isnull().sum().sum()
if missing_values_count > 0:
    print(f"Dataset contains {missing_values_count} missing values. Removing rows with missing values.")
    loan_dataset.dropna(inplace=True)
    print(f"Rows after removing missing values: {loan_dataset.shape[0]}")

# Data preprocessing
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# Handle categorical variables
loan_dataset.replace({'Married': {'No': 0, 'Yes': 1},
                      'Gender': {'Male': 1, 'Female': 0},
                      'Self_Employed': {'No': 0, 'Yes': 1},
                      'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                      'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)

# Separating the features and the target variable
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Training the Support Vector Machine model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Model evaluation
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data: ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data: ', test_data_accuracy)

# Save the trained model using joblib
joblib.dump(classifier, 'loan_model.pkl')
print("Model saved as loan_model.pkl")