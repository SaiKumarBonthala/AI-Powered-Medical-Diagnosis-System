import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv(r"C:\Users\bsaik\OneDrive\Desktop\Medical system project\diabetes_data.csv")
print(data.head())
print(data.info())
print("Missing Values:\n", data.isnull().sum())
print("Column Names:", data.columns)
X = data.drop(columns=['Outcome'], axis=1)
Y = data['Outcome']
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
model_filename = 'diabetes_model.sav'
pickle.dump(model, open(model_filename, 'wb'))
loaded_model = pickle.load(open(model_filename, 'rb'))
sample_input = (6, 148, 72, 35, 0, 33.6, 0.627, 50)
sample_array = np.asarray(sample_input).reshape(1, -1)
sample_scaled = scaler.transform(sample_array)
prediction = loaded_model.predict(sample_scaled)
if prediction[0] == 0:
    print("The Person does NOT have Diabetes.")
else:
    print("The Person HAS Diabetes.")