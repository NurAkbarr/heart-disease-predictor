import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Prediksi Penyakit Jantung", layout="wide")

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# =========================
# Sidebar
# =========================
st.sidebar.title("🫀 Form Data Pasien")

umur = st.sidebar.slider("Umur", 20, 80, 50)
gender = st.sidebar.radio("Jenis Kelamin", [0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
cp = st.sidebar.selectbox("Tipe Nyeri Dada", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Tekanan Darah", 80, 200, 120)
chol = st.sidebar.slider("Kolesterol", 100, 600, 200)
fbs = st.sidebar.radio("Gula Darah >120?", [0, 1])
restecg = st.sidebar.selectbox("Hasil EKG", [0, 1, 2])
thalach = st.sidebar.slider("Detak Jantung Maksimum", 70, 210, 150)
exang = st.sidebar.radio("Angina saat olahraga?", [0, 1])
oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope ST", [0, 1, 2])
ca = st.sidebar.selectbox("Jumlah Pembuluh Dicat", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thal", [0, 1, 2, 3])

input_data = [[umur, gender, cp, trestbps, chol, fbs, restecg,
               thalach, exang, oldpeak, slope, ca, thal]]

# =========================
# Tab Layout
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "📈 Evaluasi", "📉 Visualisasi", "🔮 Prediksi"])

# =========================
# Machine Learning
# =========================
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

# =========================
# Tab: Data
# =========================
with tab1:
    st.subheader("📋 Preview Dataset")
    st.dataframe(df)

# =========================
# Tab: Evaluasi
# =========================
with tab2:
    st.subheader("🎯 Evaluasi Model")
    st.metric("Akurasi", f"{akurasi:.2%}")
    st.write("Model: RandomForestClassifier")
    st.write(model)

# =========================
# Tab: Visualisasi
# =========================
with tab3:
    st.subheader("📊 Distribusi Target")
    fig, ax = plt.subplots()
    sns.countplot(x="target", data=df, ax=ax)
    ax.set_xticklabels(['Tidak Berisiko', 'Berisiko'])
    st.pyplot(fig)

# =========================
# Tab: Prediksi
# =========================
with tab4:
    st.subheader("🔮 Prediksi Risiko Penyakit Jantung")
    if st.button("Prediksi Sekarang"):
        hasil = model.predict(input_data)[0]
        if hasil == 1:
            st.error("⚠️ Pasien Berisiko Mengidap Penyakit Jantung.")
        else:
            st.success("✅ Pasien Tidak Berisiko Mengidap Penyakit Jantung.")

st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit")
