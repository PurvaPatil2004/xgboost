import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Credit Balance Prediction Hub",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN CUSTOM CSS STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stSidebar {
        background-color: #1e293b;
    }
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0284c7 0%, #1d4ed8 100%);
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model_path = "xgboost.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- HEADER SECTION ---
st.title("💳 Credit Balance Intelligence Dashboard")
st.markdown("Predict individual credit card balances smoothly using an advanced **XGBoost Regressor** pipeline[cite: 1]. Adjust the parameters on the sidebar to get instant predictions.")
st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header("🔍 Customer Profile Parameters")
st.sidebar.markdown("Provide the financial and demographic attributes below:")

with st.sidebar.form("prediction_form"):
    st.subheader("💰 Financial Metrics")
    income = st.number_input("Income (in $10k)", min_value=0.0, max_value=200.0, value=35.5, step=0.5, help="Annual income in tens of thousands of dollars")
    limit = st.slider("Credit Limit ($)", min_value=150, max_value=15000, value=4500, step=50)
    rating = st.slider("Credit Rating", min_value=50, max_value=1000, value=350, step=10)
    cards = st.slider("Number of Cards", min_value=1, max_value=10, value=2, step=1)

    st.subheader("👤 Demographics & Status")
    age = st.slider("Age (Years)", min_value=18, max_value=100, value=35, step=1)
    education = st.slider("Education (Years)", min_value=0, max_value=25, value=12, step=1)
    
    # Feature mappings based on standard dataset structure
    gender_map = {"Female": 0, "Male": 1}
    gender = st.selectbox("Gender", options=list(gender_map.keys()))
    
    student_map = {"No": 0, "Yes": 1}
    student = st.selectbox("Student Status", options=list(student_map.keys()))
    
    married_map = {"No": 0, "Yes": 1}
    married = st.selectbox("Marital Status", options=list(married_map.keys()))
    
    ethnicity_map = {"African American": 0, "Asian": 1, "Caucasian": 2}
    ethnicity = st.selectbox("Ethnicity", options=list(ethnicity_map.keys()))

    submitted = st.form_submit_button("🔮 Predict Credit Balance")

# --- MAIN CONTENT AREA ---
if model is None:
    st.error("⚠️ Model file `xgboost.pkl` not found in the current directory! Please upload your trained model file.")
else:
    # Prepare input dataframe matching model feature names:
    # ['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education', 'Gender', 'Student', 'Married', 'Ethnicity'][cite: 1]
    input_data = pd.DataFrame([[
        income,
        limit,
        rating,
        cards,
        age,
        education,
        gender_map[gender],
        student_map[student],
        married_map[married],
        ethnicity_map[ethnicity]
    ]], columns=['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education', 'Gender', 'Student', 'Married', 'Ethnicity'])

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📋 Active Parameter Summary")
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

    with col2:
        st.subheader("⚡ Prediction Results")
        if submitted:
            with st.spinner("Analyzing profile patterns..."):
                prediction = model.predict(input_data)[0]
            
            # Display Stylish Metric Card
            st.markdown(f"""
                <div class="metric-card">
                    <p style="color: #94a3b8; font-size: 1.1rem; margin-bottom: 5px;">Estimated Credit Balance</p>
                    <div class="metric-value">${prediction:,.2f}</div>
                    <p style="color: #38bdf8; font-size: 0.9rem; margin-top: 10px;">Model Confidence: Optimal</p>
                </div>
            """, unsafe_allow_html=True)
            
            if prediction > 1000:
                st.info("💡 **Insight:** This profile reflects a moderate-to-high credit balance utilization level.")
            else:
                st.success("💡 **Insight:** This profile reflects a healthy, lower credit balance range.")
        else:
            st.info("👈 Fill out the sidebar form and click **'Predict Credit Balance'** to view results.")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>Built with Streamlit & XGBoost • Designed for High Performance</p>", unsafe_allow_html=True)
