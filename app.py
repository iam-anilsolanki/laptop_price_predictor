import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Set page config FIRST
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# ---------- Style ----------
st.markdown("""
    <style>
    body {
        background-color: #1f1b2e;
        color: #f0f0f5;
    }
    .stApp {
        background: linear-gradient(to bottom right, #1f1b2e, #292346);
        color: #f0f0f5;
    }
    .css-1cpxqw2, .css-1v3fvcr {
        background-color: #292346 !important;
        color: #f0f0f5 !important;
    }
    label {
        color: #c8b6ff !important;   /* Soft purple for labels like Brand, OS, etc. */
        font-weight: 600;
        font-size: 14px;
    }
    .stButton > button {
        background-color: #4c3f91;
        color: white;
        border-radius: 8px;
        padding: 8px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("laptops_train.csv")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Replace categories in weight column with numeric values
    df['weight'] = df['weight'].replace({
        'ThinNlight': 2.0,
        'Casual': 2.5,
        'Gaming': 3.0,
        'Ultrabook': 1.5,
        'Notebook': 2.3,
        'Netbook': 1.2,
        'Missing': 2.4
    })

    # Extract digits from text like "4 GB GB"
    for col in ['ram_gb', 'ssd', 'hdd', 'graphic_card_gb']:
        df[col] = df[col].astype(str).str.extract(r'(\d+)').fillna(0).astype(float)

    # Convert display size to float
    df['display_size'] = pd.to_numeric(df['display_size'], errors='coerce').fillna(15.6)

    # Fix weight again in case anything slipped
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(2.3)

    return df

# ---------- Preprocessing ----------
def preprocess(df):
    df = df.copy()
    label_cols = [
        'brand', 'model', 'processor_brand', 'processor_name', 'processor_gnrtn',
        'ram_type', 'os', 'os_bit', 'warranty', 'touchscreen'
    ]
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    features = [
        'brand', 'model', 'processor_brand', 'processor_name', 'processor_gnrtn',
        'ram_gb', 'ram_type', 'ssd', 'hdd', 'os', 'os_bit',
        'graphic_card_gb', 'weight', 'display_size', 'warranty',
        'touchscreen'
    ]

    X = df[features]
    y = df['latest_price']
    return X, y, encoders, features

# ---------- Model ----------
@st.cache_resource
def train_model(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

# ---------- Load and preprocess ----------
raw_df = load_data()
X, y, encoders, features = preprocess(raw_df)
model = train_model(X, y)

# ---------- UI ----------
st.title("💻 Laptop Price Predictor")
st.write("Fill in the specifications to predict the price:")

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("Brand", encoders['brand'].classes_)
    processor_brand = st.selectbox("Processor Brand", encoders['processor_brand'].classes_)
    ram_gb = st.selectbox("RAM (GB)", sorted(raw_df['ram_gb'].unique()))
    ssd = st.selectbox("SSD (GB)", sorted(raw_df['ssd'].unique()))

with col2:
    processor_name = st.selectbox("Processor Name", encoders['processor_name'].classes_)
    processor_gnrtn = st.selectbox("Processor Generation", encoders['processor_gnrtn'].classes_)
    hdd = st.selectbox("HDD (GB)", sorted(raw_df['hdd'].unique()))
    os = st.selectbox("Operating System", encoders['os'].classes_)

with col3:
    graphic_card_gb = st.selectbox("Graphic Card (GB)", sorted(raw_df['graphic_card_gb'].unique()))
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=4.0, step=0.1)
    display_size = st.number_input("Display Size (inches)", min_value=10.0, max_value=20.0, step=0.1)
    touchscreen = st.selectbox("Touchscreen", encoders['touchscreen'].classes_)

# Warranty & OS Bit
warranty = st.selectbox("Warranty", encoders['warranty'].classes_)
os_bit = st.selectbox("OS Bit", encoders['os_bit'].classes_)

# ---------- Prediction ----------
if st.button("Predict Price"):
    input_dict = {
        'brand': brand,
        'model': 'Ideapad',  # Fixed default
        'processor_brand': processor_brand,
        'processor_name': processor_name,
        'processor_gnrtn': processor_gnrtn,
        'ram_gb': ram_gb,
        'ram_type': 'DDR4',  # Default ram type
        'ssd': ssd,
        'hdd': hdd,
        'os': os,
        'os_bit': os_bit,
        'graphic_card_gb': graphic_card_gb,
        'weight': weight,
        'display_size': display_size,
        'warranty': warranty,
        'touchscreen': touchscreen
    }

    input_df = pd.DataFrame([input_dict])

    # Encode input using trained encoders
    for col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

    # Align feature order
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹{int(prediction):,}")
