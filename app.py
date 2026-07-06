import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =====================================================================
# 1. Page Configuration & Sidebar Model Information
# =====================================================================
st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide"
)

# Sidebar - Technical Specifications Matrix
st.sidebar.header("📊 Model Information")
st.sidebar.markdown("""
**Core Architecture:**
`XGBoost Regressor`

**Validation Performance Metrics (5-Fold CV):**
* **$R^2$ Score:** `0.9885`
* **RMSE:** `₹2438`
* **MAE:** `₹1172`
""")

st.sidebar.divider()
st.sidebar.markdown("### 🛠️ Optimization Framework")
st.sidebar.caption("Hyperparameters optimized via Optuna framework. Preprocessing engineered to eliminate operational data leakage.")

# =====================================================================
# 2. Pipeline Dependencies & Robust Feature Schema Mapping
# =====================================================================
MODEL_PATH = "xgboost_flight_price.joblib"

@st.cache_resource
def load_production_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_production_model(MODEL_PATH)

# Definitive categorical levels matching the training schema exactly
airlines = ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India']
cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']

stops_map = {'0': 0, '1': 1, '2+': 2}
class_map = {'Economy': 0, 'Business': 1}

# CRITICAL FIX: Explicit hardcoded schema order from the training get_dummies configuration
# This ensures a perfect matrix shape match without relying on external file dependencies
EXPECTED_MODEL_FEATURES = [
    'stops', 'class', 'duration', 'days_left',
    'airline_AirAsia', 'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo', 'airline_SpiceJet', 'airline_Vistara',
    'source_city_Bangalore', 'source_city_Chennai', 'source_city_Delhi', 'source_city_Hyderabad', 'source_city_Kolkata', 'source_city_Mumbai',
    'departure_time_Afternoon', 'departure_time_Early_Morning', 'departure_time_Evening', 'departure_time_Late_Night', 'departure_time_Morning', 'departure_time_Night',
    'arrival_time_Afternoon', 'arrival_time_Early_Morning', 'arrival_time_Evening', 'arrival_time_Late_Night', 'arrival_time_Morning', 'arrival_time_Night',
    'destination_city_Bangalore', 'destination_city_Chennai', 'destination_city_Delhi', 'destination_city_Hyderabad', 'destination_city_Kolkata', 'destination_city_Mumbai'
]

# =====================================================================
# 3. Application Interface - Header
# =====================================================================
st.title("✈️ Flight Price Prediction")
st.markdown("""
Estimate airline ticket prices using a highly optimized XGBoost regression model trained on over 300,000 corporate flight records. Fill out the operational parameters below to generate a real-time fare projection.
""")
st.divider()

if model is None:
    st.error(f"Critical Error: Core serialization asset `{MODEL_PATH}` not found in project workspace directory. Please ensure the model file is generated before starting the UI application.")
    st.stop()

# =====================================================================
# 4. Input Configuration Form
# =====================================================================
with st.form(key='prediction_input_form'):
    st.subheader("📋 Flight Itinerary Specifications")

    # Route Domain Configuration
    col1, col2 = st.columns(2)
    with col1:
        source_city = st.selectbox("Source City", options=cities, index=0)
    with col2:
        destination_city = st.selectbox("Destination City", options=cities, index=1)

    # Operator & Class Domain Configuration
    col3, col4 = st.columns(2)
    with col3:
        airline = st.selectbox("Operating Airline", options=airlines, index=2)
    with col4:
        flight_class = st.radio("Cabin Class", options=['Economy', 'Business'], horizontal=True)

    # Temporal & Structural Vectors
    col5, col6 = st.columns(2)
    with col5:
        departure_time = st.selectbox("Departure Time Window", options=times, index=1)
    with col6:
        arrival_time = st.selectbox("Arrival Time Window", options=times, index=4)

    col7, col8 = st.columns(2)
    with col7:
        stops_input = st.radio("Number of Route Stops", options=['0', '1', '2+'], horizontal=True, index=1)
    with col8:
        duration = st.number_input("Flight Duration (Hours)", min_value=0.5, max_value=50.0, value=2.5, step=0.5)

    days_left = st.slider("Days Remaining Until Departure Date", min_value=1, max_value=49, value=20)

    # Form submission trigger
    submit_button = st.form_submit_button(label="Predict Flight Price", type="primary")

# =====================================================================
# 5. Pipeline Preprocessing, Inference & Summary Display
# =====================================================================
if submit_button:
    # Business Logic Guard: Validate that source and destination are distinct
    if source_city == destination_city:
        st.error(f"❌ Invalid Itinerary: Source city and Destination city cannot both be **{source_city}**. Please select distinct cities for your flight route.")
    else:
        # 1. Initialize a base zeroed dictionary matching all expected model feature columns
        encoded_features = {feat: 0 for feat in EXPECTED_MODEL_FEATURES}

        # 2. Map structural continuous numerical and ordinal features
        encoded_features['stops'] = stops_map[stops_input]
        encoded_features['class'] = class_map[flight_class]
        encoded_features['duration'] = float(duration)
        encoded_features['days_left'] = int(days_left)

        # 3. Handle One-Hot Encoding values dynamically by setting chosen levels to 1
        encoded_features[f'airline_{airline}'] = 1
        encoded_features[f'source_city_{source_city}'] = 1
        encoded_features[f'departure_time_{departure_time}'] = 1
        encoded_features[f'arrival_time_{arrival_time}'] = 1
        encoded_features[f'destination_city_{destination_city}'] = 1

        # 4. Convert structural map directly to DataFrame with explicit training column sorting
        df_final_features = pd.DataFrame([encoded_features])[EXPECTED_MODEL_FEATURES]

        # 5. Execute Model Inference
        predicted_fare = model.predict(df_final_features)[0]
        final_fare = max(0.0, predicted_fare) # Lower boundary protection

        # Display Result Metric Card
        st.divider()
        st.metric(
            label="Estimated Ticket Fare Prediction",
            value=f"₹{final_fare:,.2f}"
        )

        # Input Summary Parameters Panel
        st.subheader("📑 Itinerary Inference Summary")
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown(f"**Journey:** {source_city} $\\rightarrow$ {destination_city}")
            st.markdown(f"**Airline:** {airline}")
            st.markdown(f"**Cabin Class:** {flight_class}")
        with summary_col2:
            st.markdown(f"**Flight Duration:** {duration} Hours")
            st.markdown(f"**Route Layout:** {stops_input} Stop(s)")
            st.markdown(f"**Days to Departure:** {days_left} Days")

st.divider()
# =====================================================================
# 6. Interpretability & Diagnostics Reference Blocks (Static Media Assets)
# =====================================================================
st.subheader("🔍 Model Explainability")

st.markdown(
    "Explore how the trained XGBoost model makes predictions using global feature importance and SHAP explainability."
)

st.info("""
• **Feature Importance** ranks the variables that contribute most to the model.
• **SHAP Summary** explains how each feature increases or decreases the predicted flight price.
""")

tab1, tab2 = st.tabs(["Feature Importance", "SHAP Summary"])

with tab1:
    if os.path.exists("shap2.png"):
        left, center, right = st.columns([1, 4, 1])
        with center:
            st.image(
                "shap2.png",
                caption="XGBoost Feature Importance (Gain)"
            )
    else:
        st.warning("Feature importance image not found.")

with tab2:
    if os.path.exists("shap.png"):
        left, center, right = st.columns([1, 4, 1])
        with center:
            st.image(
                "shap.png",
                caption="SHAP Summary Plot"
            )
    else:
        st.warning("SHAP summary image not found.")
# =====================================================================
# 7. Page Footer
# =====================================================================
st.divider()
st.markdown("""
<div style="text-align:center; color:gray; font-size:14px;">
© 2026 Jayed Ansari<br>
Built with <b>Python</b>, <b>XGBoost</b>, <b>Optuna</b>, <b>SHAP</b>, <b>Scikit-learn</b>, and <b>Streamlit</b>.
</div>
""", unsafe_allow_html=True)
