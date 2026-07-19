import streamlit as st
import pandas as pd
import joblib

model=joblib.load('KNN_heart.pkl')
scaler=joblib.load('scaler.pkl')
ex_columns=joblib.load('columns.pkl')


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #E63946;
    text-align: center;
}

.stButton>button {
    width: 100%;
    height: 55px;
    background: linear-gradient(90deg,#ff4b4b,#ff6b6b);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg,#d62828,#ff4b4b);
}

div[data-baseweb="select"]{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<h1>❤️ Heart Disease Prediction System</h1>

<h4 style='text-align:center;color:gray;'>
AI Powered Healthcare Risk Assessment
</h4>

<hr>
""", unsafe_allow_html=True)
with st.sidebar:

    st.title("❤️ Heart Predictor")

    st.markdown("---")

    st.info("""
This application predicts the
risk of Heart Disease using
Machine Learning.

👨‍💻 Developed by DJ
""")

    st.markdown("---")

    st.success("Model : KNN")

    st.write("Python")
    st.write("Streamlit")
    st.write("Machine Learning")
left, right = st.columns(2)

with left:

    age = st.slider("👤 Age",18,100,40)

    sex = st.selectbox(
        "🚻 Gender",
        ["Male","Female"]
    )

    chest_pain = st.selectbox(
        "💓 Chest Pain",
        ['ATA','NAP','TA','ASY']
    )

    resting_bp = st.number_input(
        "🩸 Resting BP",
        80,200,120
    )

    cholesterol = st.number_input(
        "🧪 Cholesterol",
        100,600,200
    )

with right:

    fasting_bs = st.selectbox(
        "🍬 Fasting Blood Sugar",
        [0,1]
    )

    resting_ecg = st.selectbox(
        "📈 Resting ECG",
        ["Normal","ST","LVH"]
    )

    max_hr = st.slider(
        "❤️ Max Heart Rate",
        60,220,150
    )

    exercise_angina = st.selectbox(
        "🏃 Exercise Angina",
        ["Yes","No"]
    )

    oldpeak = st.slider(
        "📉 Old Peak",
        0.0,6.0,1.0
    )

    st_slope = st.selectbox(
        "📊 ST Slope",
        ["Up","Flat","Down"]
    )
    st.markdown("## 📋 Patient Summary")

c1, c2, c3 = st.columns(3)

c1.metric("👤 Age", age)
c2.metric("🩸 BP", resting_bp)
c3.metric("❤️ Max HR", max_hr)

c4, c5, c6 = st.columns(3)

c4.metric("🧪 Cholesterol", cholesterol)
c5.metric("🚻 Gender", sex)
c6.metric("💓 Chest Pain", chest_pain)

st.markdown("---")

if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1}


    
    input_df = pd.DataFrame([raw_input])

    for col in ex_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df [ex_columns]

    scaled_input = scaler.transform(input_df)
    import time

    with st.spinner("🔍 Analyzing Patient Data..."):
        time.sleep(2)
        prediction = model.predict(scaled_input)[0]

    if prediction == 1:

        st.markdown("""
        <div style="
            background:#ffe6e6;
            padding:25px;
            border-radius:15px;
            text-align:center;
            border-left:8px solid red;
        ">
            <h2 style="color:red;">⚠️ HIGH RISK</h2>
            <h4>Please consult a Cardiologist.</h4>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
            background:#e8ffe8;
            padding:25px;
            border-radius:15px;
            text-align:center;
            border-left:8px solid green;
        ">
            <h2 style="color:green;">✅ LOW RISK</h2>
            <h4>Your Heart appears Healthy.</h4>
        </div>
        """, unsafe_allow_html=True)

    chart = pd.DataFrame(
    {
        "Feature": ["BP", "Cholesterol", "Max HR"],
        "Value": [resting_bp, cholesterol, max_hr],
    }
)

   