import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="centered"
)
st.markdown("""
<style>

/* ===== GLOBAL BACKGROUND ===== */
.stApp {
    background: linear-gradient(
        135deg,
        #f0f4ff,   /* very light blue */
        #e6f7f1,   /* mint green */
        #fff7e6    /* warm cream */
    );
    background-attachment: fixed;
    padding:100px;
}
            /* ===== GAP ABOVE MAIN TITLE ===== */
h1 {
    margin-top: 2.5rem !important;   /* adjust height here */
}

            /* ===== FIX SIDEBAR SCROLL ISSUE ===== */
section[data-testid="stSidebar"] {
    height: 100vh;
    overflow-y: auto !important;
    padding-bottom: 2rem;
}

/* Ensure sidebar content does not get cut */
section[data-testid="stSidebar"] > div {
    height: auto !important;
    overflow: visible !important;
}



/* ===== MAIN CONTENT CARD EFFECT ===== */
.block-container {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

/* ===== SIDEBAR STYLING ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1f3c88,
        #2a5298,
        #1e3c72
    );
    color: white;
}



/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(
        90deg,
        #00c6ff,
        #0072ff
    );
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
}

/* ===== SUCCESS / ERROR ===== */
div.stAlert-success {
    background: linear-gradient(90deg, #e0f8e9, #c6f6d5);
    border-radius: 10px;
}

div.stAlert-error {
    background: linear-gradient(90deg, #ffe0e0, #ffbdbd);
    border-radius: 10px;
}

/* ===== HEADINGS ===== */
h1, h2, h3 {
    font-weight: 700;
}

/* ===== FOOTER ===== */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE & DESCRIPTION
# -----------------------------
st.title("🏦 Smart Loan Approval System")
st.markdown(
    """
    **This system uses Support Vector Machines (SVM) to predict loan approval.**  
    It handles non-linear decision boundaries commonly found in financial data.
    """
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("loan.csv")

df = load_data()

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Education', 'Property_Area']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# -----------------------------
# REMOVE OUTLIERS (IQR)
# -----------------------------
for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

# -----------------------------
# ENCODE CATEGORICAL VARIABLES
# -----------------------------
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = df[[
    'ApplicantIncome',
    'LoanAmount',
    'Credit_History',
    'Self_Employed',
    'Property_Area'
]]
y = df['Loan_Status']

# -----------------------------
# SPLIT & SCALE
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# SIDEBAR INPUT SECTION
# -----------------------------
st.sidebar.header("📋 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", 500, 100000, 5000)
loan_amt = st.sidebar.number_input("Loan Amount", 10, 700, 150)
credit_hist = st.sidebar.selectbox("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -----------------------------
# MODEL SELECTION
# -----------------------------
st.sidebar.header("⚙️ Model Selection")

kernel = st.sidebar.radio(
    "Choose SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

kernel_map = {
    "Linear SVM": ("linear", {}),
    "Polynomial SVM": ("poly", {"degree": 3}),
    "RBF SVM": ("rbf", {"gamma": "scale"})
}

# -----------------------------
# TRAIN MODEL
# -----------------------------
kernel_name, params = kernel_map[kernel]
model = SVC(kernel=kernel_name, C=1, **params)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("✅ Check Loan Eligibility"):

    input_df = pd.DataFrame([{
        'ApplicantIncome': app_income,
        'LoanAmount': loan_amt,
        'Credit_History': 1.0 if credit_hist == "Yes" else 0.0,
        'Self_Employed': label_encoders['Self_Employed'].transform([employment])[0],
        'Property_Area': label_encoders['Property_Area'].transform([property_area])[0]
    }])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # Confidence (distance from hyperplane)
    confidence = abs(model.decision_function(input_scaled)[0])
    confidence = min(confidence / 3, 1.0) * 100

    # -----------------------------
    # OUTPUT SECTION
    # -----------------------------
    st.markdown("---")
    st.subheader("📌 Loan Decision")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown(f"""
    **Kernel Used:** {kernel}  
    **Model Accuracy:** {accuracy:.2f}  
    **Confidence Score:** {confidence:.1f}%
    """)

    # -----------------------------
    # BUSINESS EXPLANATION
    # -----------------------------
    st.markdown("### 🧠 Business Explanation")

    if prediction == 1:
        st.info(
            "Based on the applicant’s income pattern and positive credit history, "
            "the model predicts a **high likelihood of loan repayment**."
        )
    else:
        st.warning(
            "Based on income level and credit history patterns, "
            "the applicant is **less likely to repay the loan**."
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("SVM-based FinTech System")
