import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv(r"C:\Users\bsaik\OneDrive\Desktop\Medical system project\heart_disease_data.csv")
print(data.head())
print(data.info())
print("Missing Values:\n", data.isnull().sum())
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
data['thal'] = pd.to_numeric(data['thal'], errors='coerce')
data.dropna(inplace=True)
print("Column Names:", data.columns)
X = data.drop(columns=['target'], axis=1)
Y = data['target']
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
model_filename = 'heart_disease_model.sav'
pickle.dump(model, open(model_filename, 'wb'))
loaded_model = pickle.load(open(model_filename, 'rb'))
sample_input = (63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1)
sample_array = np.asarray(sample_input).reshape(1, -1)
sample_scaled = scaler.transform(sample_array)
prediction = loaded_model.predict(sample_scaled)
if prediction[0] == 0:
    print("The Person does NOT have Heart Disease.")
else:
    print("The Person HAS Heart Disease.")