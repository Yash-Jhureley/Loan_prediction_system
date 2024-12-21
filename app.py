# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('loan_model.pkl')

# Streamlit app title
st.title("Loan Prediction System")

# Create input fields for user data
st.header("Enter your loan application details")

# Input fields
gender = st.selectbox("Gender", ["Select Gender", "Male", "Female"])
married = st.selectbox("Married", ["Select Status", "Yes", "No"])
dependents = st.selectbox("Dependents", ["Select Dependents", "0", "1", "2", "3+"])
education = st.selectbox("Education", ["Select Education", "Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Select Status", "Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)
credit_history = st.selectbox("Credit History", ["Select History", "Yes", "No"])
property_area = st.selectbox("Property Area", ["Select Area", "Rural", "Semiurban", "Urban"])

# Button to make prediction
if st.button("Predict"):
    # Check for missing values in input fields
    if (gender == "Select Gender" or
        married == "Select Status" or
        dependents == "Select Dependents" or
        education == "Select Education" or
        self_employed == "Select Status" or
        credit_history == "Select History" or
        property_area == "Select Area"):
        st.error("Please fill all fields before making a prediction.")
    else:
        # Prepare input data for prediction
        input_data = [[gender, married, dependents, education, self_employed, applicant_income, coapplicant_income,
                       loan_amount, loan_amount_term, credit_history, property_area]]

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data, columns=['Gender', 'Married', 'Dependents', 'Education', 
                                                      'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
                                                      'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
                                                      'Property_Area'])

        # Preprocess the input data
        input_df.replace({'Married': {'No': 0, 'Yes': 1},
                          'Gender': {'Male': 1, 'Female': 0},
                          'Self_Employed': {'No': 0, 'Yes': 1},
                          'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                          'Education': {'Graduate': 1, 'Not Graduate': 0},
                          'Credit_History': {'No': 0, 'Yes': 1}}, inplace=True)
        input_df['Dependents'] = input_df['Dependents'].replace('3+', 3)

        # Debugging: Check the input DataFrame
        st.write("Input Data for Prediction:")
        st.dataframe(input_df)

        # Make prediction
        try:
            prediction = model.predict(input_df)

            # Debugging: Show the raw prediction output
            st.write("Raw Prediction Output:", prediction)

            # Display the prediction result
            if prediction[0] == 1:
                st.success("Loan Approved")
            elif prediction[0]==0:
                st.error("Loan Rejected")

            

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error(f"An error occurred during prediction: {e}")