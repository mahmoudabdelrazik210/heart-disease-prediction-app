import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")
st.markdown("## 🫀 Heart Disease Prediction App")
st.markdown("Enter the patient's details below:")

@st.cache_resource
def load_model():
    return joblib.load("best_xgboost_model.pkl")

model = load_model()

st.markdown("### 🧍‍♂️ Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age | العمر (Normal: 20–60 years)", 1, 120, 50)
    sex = st.selectbox("🚻 Gender | الجنس (0 = Female | 1 = Male)", [0, 1], format_func=lambda x: "أنثى (Female)" if x == 0 else "ذكر (Male)")
    cp = st.selectbox("💔 Chest Pain Type | نوع ألم الصدر (0=Normal → 3=Worst)", [0, 1, 2, 3], format_func=lambda x: {0: "0 - Typical Angina | ذبحة صدرية طبيعية", 1: "1 - Atypical Angina | ذبحة غير نموذجية", 2: "2 - Non-anginal Pain | ألم غير قلبي", 3: "3 - Asymptomatic | بدون أعراض"}[x])
    trestbps = st.number_input("🩺 Resting BP | ضغط الدم أثناء الراحة (Normal: 90–120 mmHg)", 80, 200, 120)
    chol = st.number_input("💉 Serum Cholesterol | الكوليسترول (Normal: < 200 mg/dl)", 100, 600, 200)
    fbs = st.selectbox("🩸 Fasting Blood Sugar | السكر الصائم (0 <120 mg/dl | 1 >120 mg/dl)", [0, 1], format_func=lambda x: "0 - طبيعي (<120)" if x == 0 else "1 - مرتفع (>120)")

with col2:
    restecg = st.selectbox("📈 Resting ECG | تخطيط القلب أثناء الراحة (0=Normal)", [0, 1, 2], format_func=lambda x: {0: "0 - طبيعي (Normal)", 1: "1 - ST-T abnormality (خلل بسيط)", 2: "2 - LVH (تضخم البطين الأيسر)"}[x])
    thalch = st.number_input("❤️ Max Heart Rate | أقصى معدل نبض (Normal: 100–200 bpm)", 60, 220, 150)
    exang = st.selectbox("🏃 Exercise Angina | ألم مع التمرين (0 = No | 1 = Yes)", [0, 1], format_func=lambda x: "0 - لا يوجد (No)" if x == 0 else "1 - يوجد (Yes)")
    oldpeak = st.number_input("📉 Oldpeak (ST Depression) | انخفاض ST (Normal: 0.0–2.0)", 0.0, 6.0, 1.0)
    slope = st.selectbox("📊 Slope of ST Segment | ميل منحنى ST (0 = Upsloping أفضل)", [0, 1, 2], format_func=lambda x: {0: "0 - صاعد (Upsloping)", 1: "1 - مستوٍ (Flat)", 2: "2 - نازل (Downsloping)"}[x])
    ca = st.selectbox("🫀 Major Vessels (CA) | عدد الشرايين المسدودة (0 = سليم)", [0, 1, 2, 3], format_func=lambda x: f"{x} - {'سليم' if x == 0 else f'{x} شريان متأثر'}")
    thal = st.selectbox("🧬 Thalassemia | الثلاسيميا (0 = Normal)", [0, 1, 2], format_func=lambda x: {0: "0 - طبيعي (Normal)", 1: "1 - خلل ثابت (Fixed Defect)", 2: "2 - خلل متغير (Reversible Defect)"}[x])

if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100
    st.markdown("### 📊 Prediction Result | النتيجة")
    if prob < 40:
        st.success(f"🟩 Low Risk ({prob:.2f}%) — خطر منخفض | المريض بصحة جيدة غالبًا")
        color = "#10B981"
    elif prob <= 70:
        st.warning(f"🟧 Medium Risk ({prob:.2f}%) — خطر متوسط | يُفضل عمل فحوصات إضافية")
        color = "#FBBF24"
    else:
        st.error(f"🟥 High Risk ({prob:.2f}%) — خطر مرتفع | يُنصح بمراجعة طبيب القلب فورًا")
        color = "#EF4444"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=prob, title={"text": "Heart Disease Risk (%) | نسبة خطر أمراض القلب", "font": {"size": 18}}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}, "steps": [{"range": [0, 40], "color": "#6EE7B7"}, {"range": [40, 70], "color": "#FCD34D"}, {"range": [70, 100], "color": "#F87171"}], "threshold": {"line": {"color": "black", "width": 4}, "value": prob}}))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("👨‍💻 Developed by Kerolos Medhat | Heart Disease Prediction App")
