import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
data = pd.read_csv(r"C:\Users\bsaik\OneDrive\Desktop\Medical system project\survey lung cancer.csv")

# Encode categorical variables
label_encoder = LabelEncoder()
data['GENDER'] = label_encoder.fit_transform(data['GENDER'])
data['LUNG_CANCER'] = label_encoder.fit_transform(data['LUNG_CANCER'])

# Split features and target
X = data.drop(columns='LUNG_CANCER', axis=1)
Y = data['LUNG_CANCER']

# Split data into train & test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Apply Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=500, solver='lbfgs')
model.fit(X_train_scaled, Y_train)

# Save model and scaler
pickle.dump(model, open('lungs_disease_model.sav', 'wb'))
pickle.dump(scaler, open('scaler.sav', 'wb'))

# Load the saved model and scaler
loaded_model = pickle.load(open('lungs_disease_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# ----------------- Streamlit UI ----------------- #

st.title("Lung Cancer Prediction System 🫁")

st.write("""
### Enter the details below to check for lung cancer risk
""")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, step=1)
smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
yellow_fingers = st.selectbox("Do you have yellow fingers?", ["No", "Yes"])
anxiety = st.selectbox("Do you have anxiety?", ["No", "Yes"])
peer_pressure = st.selectbox("Are you influenced by peer pressure?", ["No", "Yes"])
chronic_disease = st.selectbox("Do you have any chronic disease?", ["No", "Yes"])
fatigue = st.selectbox("Do you experience fatigue?", ["No", "Yes"])
allergy = st.selectbox("Do you have allergies?", ["No", "Yes"])
wheezing = st.selectbox("Do you wheeze?", ["No", "Yes"])
alcohol = st.selectbox("Do you consume alcohol frequently?", ["No", "Yes"])
coughing = st.selectbox("Do you have a persistent cough?", ["No", "Yes"])
shortness_of_breath = st.selectbox("Do you experience shortness of breath?", ["No", "Yes"])
swallowing_difficulty = st.selectbox("Do you have difficulty swallowing?", ["No", "Yes"])
chest_pain = st.selectbox("Do you experience chest pain?", ["No", "Yes"])

# Encode inputs
gender_encoded = 1 if gender == "Male" else 0
smoking_encoded = 1 if smoking == "Yes" else 0
yellow_fingers_encoded = 1 if yellow_fingers == "Yes" else 0
anxiety_encoded = 1 if anxiety == "Yes" else 0
peer_pressure_encoded = 1 if peer_pressure == "Yes" else 0
chronic_disease_encoded = 1 if chronic_disease == "Yes" else 0
fatigue_encoded = 1 if fatigue == "Yes" else 0
allergy_encoded = 1 if allergy == "Yes" else 0
wheezing_encoded = 1 if wheezing == "Yes" else 0
alcohol_encoded = 1 if alcohol == "Yes" else 0
coughing_encoded = 1 if coughing == "Yes" else 0
shortness_of_breath_encoded = 1 if shortness_of_breath == "Yes" else 0
swallowing_difficulty_encoded = 1 if swallowing_difficulty == "Yes" else 0
chest_pain_encoded = 1 if chest_pain == "Yes" else 0

# Prepare input data
input_data = np.array([
    gender_encoded, age, smoking_encoded, yellow_fingers_encoded, anxiety_encoded, 
    peer_pressure_encoded, chronic_disease_encoded, fatigue_encoded, allergy_encoded, 
    wheezing_encoded, alcohol_encoded, coughing_encoded, shortness_of_breath_encoded, 
    swallowing_difficulty_encoded, chest_pain_encoded
]).reshape(1, -1)

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    prediction = loaded_model.predict(input_data_scaled)
    result = "The person has Lung Cancer 🫁" if prediction[0] == 1 else "The person does NOT have Lung Cancer ✅"
    st.subheader(result)
