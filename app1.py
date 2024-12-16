
# %%writefile app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import tempfile

# Streamlit app title
st.title("Loan Prediction")

# File uploader for user to upload a CSV file
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

@st.cache_data
def load_data(file):
    """Load and preprocess the dataset."""
    loan_dataset = pd.read_csv(file)
    loan_dataset = loan_dataset.dropna()
    loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
    loan_dataset.replace(to_replace='3+', value=4, inplace=True)
    loan_dataset.replace({'Married': {'No': 0, 'Yes': 1},
                          'Gender': {'Male': 1, 'Female': 0},
                          'Self_Employed': {'No': 0, 'Yes': 1},
                          'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                          'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)
    return loan_dataset

if uploaded_file is not None:
    # Use a temporary file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name
        
        # Load the dataset into a DataFrame
        try:
            loan_dataset = load_data(file_path)
            st.write("Dataset loaded successfully!")
            st.dataframe(loan_dataset.head())  # Display the first few rows of the dataset
            
            # Separate features and target
            X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
            Y = loan_dataset['Loan_Status']

            # Train-test split
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

            # Cache the model training
            @st.cache_resource
            def train_model(X_train, Y_train):
                classifier = svm.SVC(kernel='linear')
                classifier.fit(X_train, Y_train)
                return classifier

            classifier = train_model(X_train, Y_train)

            # Evaluate the model
            Y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(Y_test, Y_pred)
            st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

            # Input fields for user
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            applicant_income = st.number_input("Applicant Income", min_value=0)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0)
            loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)
            credit_history = st.selectbox("Credit History", ["Yes", "No"])
            property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

            # Convert user input to model input format
            input_data = [[gender, married, dependents, education, self_employed, applicant_income, coapplicant_income,
                           loan_amount, loan_amount_term, credit_history, property_area]]

            # Replace categorical values with numerical values
            input_data = pd.DataFrame(input_data, columns=X.columns)
            input_data.replace({'Married': {'No': 0, 'Yes': 1},
                                'Gender': {'Male': 1, 'Female': 0},
                                'Self_Employed': {'No': 0, 'Yes': 1},
                                'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                                'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)
            input_data['Dependents'] = input_data['Dependents'].replace('3+', 4)

            # Make prediction
            if st.button("Predict"):
                prediction = classifier.predict(input_data)
                if prediction[0] == 1:
                    st.success("Loan Approved")
                else:
                    st.error("Loan Rejected")

            # Reset functionality
            if st.button("Reset"):
                st.experimental_rerun()

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        except pd.errors.ParserError:
            st.error("Error parsing the file. Please ensure it is a valid CSV format.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Please upload a CSV file to get started.")