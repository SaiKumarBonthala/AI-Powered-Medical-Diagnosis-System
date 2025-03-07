import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv(r"C:\Users\bsaik\OneDrive\Desktop\Medical system project\hypothyroid.csv")
print("Available columns in dataset:", list(data.columns))
data.replace("?", np.nan, inplace=True)
if 'sex' in data.columns:
    data['sex'] = data['sex'].str.lower().map({'m': 1, 'f': 0})
label_encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = label_encoder.fit_transform(data[col])
for col in data.columns:
    if data[col].dtype in ['int64', 'float64']:
        data[col].fillna(data[col].mean(), inplace=True)
selected_features = ['age', 'sex', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
if not all(feature in data.columns for feature in selected_features):
    missing_features = [feature for feature in selected_features if feature not in data.columns]
    raise KeyError(f"❌ ERROR: Missing columns in dataset: {missing_features}")

X = data[selected_features]
Y = data['binaryClass']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)
Y_train_pred = model.predict(X_train_scaled)
Y_test_pred = model.predict(X_test_scaled)
train_accuracy = accuracy_score(Y_train, Y_train_pred)
test_accuracy = accuracy_score(Y_test, Y_test_pred)

print(f"✅ Training Accuracy: {train_accuracy:.2f}")
print(f"✅ Test Accuracy: {test_accuracy:.2f}")
print("\nClassification Report:\n", classification_report(Y_test, Y_test_pred))
model_filename = 'thyroid_disease_model.sav'
pickle.dump(model, open(model_filename, 'wb'))
loaded_model = pickle.load(open(model_filename, 'rb'))
sample_input = (45, 1, 2.5, 1.9, 110, 0.92, 102)
sample_array = np.asarray(sample_input).reshape(1, -1)
sample_scaled = scaler.transform(sample_array)
prediction = loaded_model.predict(sample_scaled)
if prediction[0] == 0:
    print("🟢 The Person does NOT have Thyroid Disease.")
else:
    print("🔴 The Person HAS Thyroid Disease.")



