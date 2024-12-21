# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('dataset.csv')

# Step 1: Handle Missing Values
# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Remove rows with missing values
data.dropna(inplace=True)

# Verify that there are no missing values left
print("Missing values after dropping rows:")
print(data.isnull().sum())

# Step 2: Encode Categorical Variables
# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Married'] = label_encoder.fit_transform(data['Married'])
data['Education'] = label_encoder.fit_transform(data['Education'])
data['Self_Employed'] = label_encoder.fit_transform(data['Self_Employed'])
data['Credit_History'] = label_encoder.fit_transform(data['Credit_History'])
data['Property_Area'] = label_encoder.fit_transform(data['Property_Area'])
data['Dependents'] = data['Dependents'].replace('3+', 4)  # Convert '3+' to 4

# Step 3: Prepare Data for Training
# Define features and target variable
X = data.drop(columns=['Loan_Status'])
y = data['Loan_Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model
model = SVC()
model.fit(X_train, y_train)

# Step 5: Make Predictions and Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of the SVM model: {accuracy * 100:.2f}%")