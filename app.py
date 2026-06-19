
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, pred)

# UI
st.title("🩺 Diabetes Prediction System")

menu = st.sidebar.selectbox("Menu", ["Predict", "Model Comparison"])

# ---------------- PREDICTION ----------------
if menu == "Predict":
    st.subheader("Enter Patient Details")

    pregnancies = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose")
    bp = st.number_input("Blood Pressure")
    skin = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_data = scaler.transform(input_data)

    model_choice = st.selectbox("Select Model", list(models.keys()))

    if st.button("Predict"):
        prediction = models[model_choice].predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Diabetic Detected")
        else:
            st.success("✅ Non-Diabetic")

# ---------------- DASHBOARD ----------------
if menu == "Model Comparison":
    st.subheader("Model Accuracy Comparison")

    st.bar_chart(results)

    st.write("### Accuracy Scores")
    for k, v in results.items():
        st.write(f"{k}: {v:.2f}")