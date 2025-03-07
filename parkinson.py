import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv(r"C:\Users\bsaik\OneDrive\Desktop\Medical system project\parkinson_data.csv")
print(data.head())
print(data.info())
print(data.describe())
print("Missing Values:\n", data.isnull().sum())
print("Column Names:", data.columns)
X = data.drop(columns=['name', 'status'], axis=1)
Y = data['status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, Y_train)
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)
train_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:\n", classification_report(Y_test, Y_test_pred))
model_filename = 'parkinsons_disease_model.sav'
pickle.dump(model, open(model_filename, 'wb'))
loaded_model = pickle.load(open(model_filename, 'rb'))
sample_input = (120.0, 145.0, 0.0023, 0.0004, 0.0021, 0.0032, 0.0019, 22.0, 215.0, 0.45, 0.89, 0.23, 0.56, 5.0, 6.0, 0.78, 0.67, 0.92, 0.0012, 0.00034, 1.9, 2.3)  # Adjust based on dataset columns
sample_array = np.asarray(sample_input).reshape(1, -1)
sample_scaled = scaler.transform(sample_array)
prediction = loaded_model.predict(sample_scaled)
if prediction[0] == 0:
    print("The Person does NOT have Parkinson's Disease.")
else:
    print("The Person HAS Parkinson's Disease.")