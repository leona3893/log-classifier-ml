""" import streamlit as st
import joblib
import re

model = joblib.load("model/log_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def clean_log(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

st.title("Log Classification Tool")

log_input = st.text_area("Nhập log hệ thống:")

if st.button("Phân loại"):
    clean = clean_log(log_input)
    vec = vectorizer.transform([clean])
    result = model.predict(vec)[0]
    st.success(f"Kết quả: {result}") """
import streamlit as st
import pickle
import pandas as pd

# Load model & vectorizer
with open("model/log_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="Log Classification Tool", layout="wide")

st.title("🔍 Log Classification Tool")
st.write("Phân loại log hệ thống bằng Machine Learning")

# =========================
# CHỌN CÁCH NHẬP LOG
# =========================
input_method = st.radio(
    "Chọn cách nhập log:",
    ("Paste log", "Upload file log")
)

logs = []

# =========================
# CÁCH 1: PASTE LOG
# =========================
if input_method == "Paste log":
    log_text = st.text_area(
        "Dán log vào đây (mỗi dòng là một log):",
        height=200
    )

    if log_text:
        logs = log_text.splitlines()

# =========================
# CÁCH 2: UPLOAD FILE
# =========================
if input_method == "Upload file log":
    uploaded_file = st.file_uploader(
        "Upload file log (.txt, .log)",
        type=["txt", "log"]
    )

    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
        logs = content.splitlines()

# =========================
# PHÂN LOẠI LOG
# =========================
if logs:
    st.subheader("📊 Kết quả phân loại")

    X = vectorizer.transform(logs)
    predictions = model.predict(X)

    df_result = pd.DataFrame({
        "Log": logs,
        "Loại": predictions
    })

    st.dataframe(df_result, use_container_width=True)

    # Thống kê
    st.subheader("📈 Thống kê")
    st.bar_chart(df_result["Loại"].value_counts())

else:
    st.info("👉 Nhập hoặc upload log để bắt đầu phân tích.")

