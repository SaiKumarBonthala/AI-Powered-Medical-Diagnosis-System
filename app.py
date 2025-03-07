import streamlit as st
import pickle

st.set_page_config(page_title="Disease Prediction", page_icon="⚕️")

# Hide Streamlit default UI
hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Background Image
background_image_url = "https://ehealth4everyone.com/wp-content/uploads/2023/01/Blog-Header-1200x600-px-3.png"
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url({background_image_url});
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stAppViewContainer"]::before {{
content: "";
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
background-color: rgba(0, 0, 0, 0.7);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load Models
models = {
    'lungs_cancer': pickle.load(open(r'C:\Users\bsaik\OneDrive\Desktop\Medical system project\lungs_disease_model.sav', 'rb')),
    'diabetes': pickle.load(open(r'C:\Users\bsaik\OneDrive\Desktop\Medical system project/diabetes_model.sav', 'rb')),
    'heart_disease': pickle.load(open(r'C:\Users\bsaik\OneDrive\Desktop\Medical system project/heart_disease_model.sav', 'rb')),
    'parkinsons': pickle.load(open(r'C:\Users\bsaik\OneDrive\Desktop\Medical system project/parkinsons_disease_model.sav', 'rb')),
    'thyroid': pickle.load(open(r'C:\Users\bsaik\OneDrive\Desktop\Medical system project/Thyroid_disease_model.sav', 'rb'))
}

# Sidebar Menu
selected = st.sidebar.radio(
    "Select a Disease to Predict",
    ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Lung Cancer Prediction', 'Hypo-Thyroid Prediction']
)

def display_input(label, tooltip, key, type="number"):
    return st.number_input(label, key=key, help=tooltip, step=1)

### **Diabetes Prediction**
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction')
    Pregnancies = display_input('Number of Pregnancies', 'Enter number of times pregnant', 'Pregnancies')
    Glucose = display_input('Glucose Level', 'Enter glucose level', 'Glucose')
    BloodPressure = display_input('Blood Pressure', 'Enter blood pressure', 'BloodPressure')
    SkinThickness = display_input('Skin Thickness', 'Enter skin thickness', 'SkinThickness')
    Insulin = display_input('Insulin Level', 'Enter insulin level', 'Insulin')
    BMI = display_input('BMI', 'Enter Body Mass Index', 'BMI')
    DiabetesPedigreeFunction = display_input('Diabetes Pedigree Function', 'Enter diabetes pedigree function', 'DiabetesPedigreeFunction')
    Age = display_input('Age', 'Enter age', 'Age')

    if st.button('Predict Diabetes'):
        prediction = models['diabetes'].predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        st.success('The person is diabetic' if prediction[0] == 1 else 'The person is not diabetic')

### **Heart Disease Prediction**
if selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction')
    age = display_input('Age', 'Enter age', 'age')
    sex = display_input('Sex (1=Male, 0=Female)', 'Enter sex', 'sex')
    cp = display_input('Chest Pain Type (0-3)', 'Enter chest pain type', 'cp')
    trestbps = display_input('Resting Blood Pressure', 'Enter blood pressure', 'trestbps')
    chol = display_input('Serum Cholesterol', 'Enter cholesterol level', 'chol')
    fbs = display_input('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', 'Enter FBS', 'fbs')
    restecg = display_input('Resting ECG (0-2)', 'Enter ECG result', 'restecg')
    thalach = display_input('Max Heart Rate', 'Enter max heart rate', 'thalach')
    exang = display_input('Exercise Induced Angina (1=Yes, 0=No)', 'Enter value', 'exang')
    oldpeak = display_input('ST Depression', 'Enter ST depression value', 'oldpeak')
    slope = display_input('Slope (0-2)', 'Enter slope value', 'slope')
    ca = display_input('Major Vessels (0-3)', 'Enter vessel count', 'ca')
    thal = display_input('Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect)', 'Enter thal', 'thal')

    if st.button('Predict Heart Disease'):
        prediction = models['heart_disease'].predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        st.success('The person has heart disease' if prediction[0] == 1 else 'The person does not have heart disease')

### **Parkinson's Prediction**
if selected == "Parkinsons Prediction":
    st.title("🧠 Parkinson's Disease Prediction")
    features = [display_input(f'Feature {i+1}', f'Enter value for Feature {i+1}', f'feature_{i+1}') for i in range(22)]

    if st.button("Predict Parkinson's"):
        prediction = models['parkinsons'].predict([features])
        st.success("The person has Parkinson's disease" if prediction[0] == 1 else "The person does not have Parkinson's disease")

### **Lung Cancer Prediction**
if selected == "Lung Cancer Prediction":
    st.title("🫁 Lung Cancer Prediction")
    features = [display_input(f'Feature {i+1}', f'Enter value for Feature {i+1}', f'feature_{i+1}') for i in range(15)]

    if st.button("Predict Lung Cancer"):
        prediction = models['lungs_cancer'].predict([features])
        st.success("The person has lung cancer" if prediction[0] == 1 else "The person does not have lung cancer")

### **Hypo-Thyroid Prediction**
if selected == "Hypo-Thyroid Prediction":
    st.title("🦠 Hypo-Thyroid Prediction")
    age = display_input('Age', 'Enter age', 'age')
    sex = display_input('Sex (1=Male, 0=Female)', 'Enter sex', 'sex')
    on_thyroxine = display_input('On Thyroxine (1=Yes, 0=No)', 'Enter value', 'on_thyroxine')
    tsh = display_input('TSH Level', 'Enter TSH level', 'tsh')
    t3_measured = display_input('T3 Measured (1=Yes, 0=No)', 'Enter value', 't3_measured')
    t3 = display_input('T3 Level', 'Enter T3 level', 't3')
    tt4 = display_input('TT4 Level', 'Enter TT4 level', 'tt4')

    if st.button("Predict Thyroid"):
        prediction = models['thyroid'].predict([[age, sex, on_thyroxine, tsh, t3_measured, t3, tt4]])
        st.success("The person has Hypo-Thyroid disease" if prediction[0] == 1 else "The person does not have Hypo-Thyroid disease")
















