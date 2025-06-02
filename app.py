import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="centered")

@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

st.title("🫀 Prediksi Risiko Penyakit Jantung")
st.markdown("Masukkan data berikut:")

age = st.slider("Umur", 20, 80, 50)
sex = st.radio("Jenis Kelamin", [0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
cp = st.selectbox("Tipe Nyeri Dada", [0, 1, 2, 3])
trestbps = st.slider("Tekanan Darah", 80, 200, 120)
chol = st.slider("Kolesterol", 100, 600, 200)
fbs = st.radio("Gula Darah >120?", [0, 1])
restecg = st.selectbox("Hasil EKG", [0, 1, 2])
thalach = st.slider("Detak Jantung Maks.", 70, 210, 150)
exang = st.radio("Angina Saat Olahraga", [0, 1])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope ST", [0, 1, 2])
ca = st.selectbox("Jumlah Pembuluh Dicat", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

input_data = [[age, sex, cp, trestbps, chol, fbs, restecg,
               thalach, exang, oldpeak, slope, ca, thal]]

if st.button("🔍 Prediksi Sekarang"):
    hasil = model.predict(input_data)[0]
    st.subheader("🔎 Hasil Prediksi")
    if hasil == 1:
        st.error("⚠️ Berisiko Mengidap Penyakit Jantung.")
    else:
        st.success("✅ Tidak Berisiko.")

st.metric("🎯 Akurasi Model", f"{acc:.2%}")
