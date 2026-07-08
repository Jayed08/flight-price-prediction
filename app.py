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
st.sidebar.header("📊 Model Overview")

st.sidebar.markdown("""
**Model:** XGBoost Regressor

**Dataset**
- 300,153 flight records
- 37 engineered features

### 🎯 Test Set Performance

- **R²:** `0.9895`
- **RMSE:** `₹2,330`
- **MAE:** `₹1,085`

### 🔁 5-Fold Cross Validation

- **R²:** `0.9889`
- **RMSE:** `₹2,395`
- **MAE:** `₹1,133`
""")

st.sidebar.divider()

st.sidebar.markdown("### ⚙️ Training Pipeline")

st.sidebar.caption("""
• Hyperparameter tuning with Optuna
• 5-fold cross-validation
• SHAP explainability
• Leakage-aware preprocessing
""")

# =====================================================================
# 2. Pipeline Dependencies & Robust Feature Schema Mapping
# =====================================================================
MODEL_PATH = "model/xgb_flight_price.pkl" if os.path.exists("model/xgb_flight_price.pkl") else "xgb_flight_price.pkl"
FEATURE_COLS_PATH = "model/feature_columns.pkl" if os.path.exists("model/feature_columns.pkl") else "feature_columns.pkl"

@st.cache_resource
def load_production_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_feature_columns(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_production_model(MODEL_PATH)
dynamic_features = load_feature_columns(FEATURE_COLS_PATH)

# Definitive categorical levels matching the training schema exactly
airlines = ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India']
cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
times = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']

# Fallback schema order matching pd.get_dummies configuration in model.py
if dynamic_features is not None:
    EXPECTED_MODEL_FEATURES = dynamic_features
else:
    EXPECTED_MODEL_FEATURES = [
        'duration', 'days_left',
        'airline_AirAsia', 'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo', 'airline_SpiceJet', 'airline_Vistara',
        'source_city_Bangalore', 'source_city_Chennai', 'source_city_Delhi', 'source_city_Hyderabad', 'source_city_Kolkata', 'source_city_Mumbai',
        'departure_time_Afternoon', 'departure_time_Early_Morning', 'departure_time_Evening', 'departure_time_Late_Night', 'departure_time_Morning', 'departure_time_Night',
        'stops_one', 'stops_two_or_more', 'stops_zero',
        'arrival_time_Afternoon', 'arrival_time_Early_Morning', 'arrival_time_Evening', 'arrival_time_Late_Night', 'arrival_time_Morning', 'arrival_time_Night',
        'destination_city_Bangalore', 'destination_city_Chennai', 'destination_city_Delhi', 'destination_city_Hyderabad', 'destination_city_Kolkata', 'destination_city_Mumbai',
        'class_Business', 'class_Economy'
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
    st.error("❌ **Critical Error: Trained Model File Not Found**")
    st.markdown(f"""
    The serialized model file (`{MODEL_PATH}`) could not be located in the project directory. 
    
    To generate the model, please run the training pipeline script:
    ```bash
    python model.py
    ```
    This will execute the model training, optimization, and validation process, saving the champion XGBoost model to `model/xgb_flight_price.pkl`.
    """)
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
        st.error(
    f"❌ Source and destination cannot both be **{source_city}**. Please choose different cities."
)
    else:
        # 1. Initialize a base zeroed dictionary matching all expected model feature columns
        encoded_features = {feat: 0 for feat in EXPECTED_MODEL_FEATURES}

        # 2. Map structural continuous numerical features
        if 'duration' in encoded_features:
            encoded_features['duration'] = float(duration)
        if 'days_left' in encoded_features:
            encoded_features['days_left'] = int(days_left)

        # 3. Handle One-Hot Encoding values dynamically by setting chosen levels to 1
        def set_feature_active(prefix, value):
            col_name = f"{prefix}_{value}"
            if col_name in encoded_features:
                encoded_features[col_name] = 1

        set_feature_active('airline', airline)
        set_feature_active('source_city', source_city)
        set_feature_active('departure_time', departure_time)
        set_feature_active('arrival_time', arrival_time)
        set_feature_active('destination_city', destination_city)

        # Map stops (UI options: '0', '1', '2+') to one-hot categories ('zero', 'one', 'two_or_more')
        stops_mapping = {'0': 'zero', '1': 'one', '2+': 'two_or_more'}
        set_feature_active('stops', stops_mapping.get(stops_input, 'zero'))

        # Map class (UI options: 'Economy', 'Business') to one-hot categories ('Economy', 'Business')
        set_feature_active('class', flight_class)

        # 4. Convert structural map directly to DataFrame with explicit training column sorting
        df_final_features = pd.DataFrame([encoded_features])[EXPECTED_MODEL_FEATURES]

        # 5. Execute Model Inference
        predicted_fare = model.predict(df_final_features)[0]
        final_fare = max(0.0, predicted_fare) # Lower boundary protection

        # Display Result Metric Card
        st.divider()
        st.success("🎉 Prediction Generated Successfully!")
        
        col_metric, col_summary = st.columns([1, 1])
        with col_metric:
            st.metric(
                label="Estimated Ticket Fare",
                value=f"₹{round(final_fare):,}",
                help="Fare estimated via the tuned XGBoost model."
            )
            st.caption("ℹ️ Estimates are projections based on historical data and dynamic flight features.")

        # Input Summary Parameters Panel
        with col_summary:
            st.subheader("📑 Itinerary Details")
            st.markdown(f"**Journey:** {source_city} → {destination_city}")
            st.markdown(f"**Airline:** {airline}")
            st.markdown(f"**Class:** {flight_class}")
            st.markdown(f"**Duration:** {duration} Hours | **Stops:** {stops_input}")
            st.markdown(f"**Days to Departure:** {days_left} Days")

st.divider()
# =====================================================================
# 6. Interpretability & Diagnostics Reference Blocks (Static Media Assets)
# =====================================================================
st.subheader("🔍 Model Explainability & Diagnostics")

st.markdown(
    "Explore how the final XGBoost model makes predictions using feature importance, SHAP explainability, and residual diagnostics."
)

st.info("""
• **Feature Importance** ranks the most influential features using XGBoost's gain metric.
• **SHAP Explanations** illustrate the direction and magnitude of feature impacts on fares.
• **Diagnostics & Residuals** display model accuracy and verify the error distribution.
""")

# Setup visual tabs matching the artifacts generated by model.py
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Feature Importance", 
    "🔮 SHAP Summary",
    "📊 Prediction Accuracy",
    "📉 Residual Diagnostics"
])

# Helper function to find a file in assets or root directory
def get_asset_path(filename):
    assets_path = os.path.join("assets", filename)
    if os.path.exists(assets_path):
        return assets_path
    elif os.path.exists(filename):
        return filename
    return None

with tab1:
    importance_img = get_asset_path("feature_importance.png")
    if importance_img:
        st.image(importance_img, use_container_width=True, caption="XGBoost Gain-Based Feature Importance")
    else:
        st.warning("Feature importance image not found.")

with tab2:
    shap_summary_img = get_asset_path("shap_summary.png")
    shap_importance_img = get_asset_path("shap_importance.png")
    
    col_shap1, col_shap2 = st.columns(2)
    with col_shap1:
        if shap_summary_img:
            st.image(shap_summary_img, use_container_width=True, caption="SHAP Summary Density Plot")
        else:
            st.warning("SHAP Summary image not found.")
    with col_shap2:
        if shap_importance_img:
            st.image(shap_importance_img, use_container_width=True, caption="Mean Absolute SHAP Values")
        else:
            st.warning("SHAP Importance image not found.")

with tab3:
    performance_img = get_asset_path("predicted_vs_actual.png")
    if performance_img:
        st.image(performance_img, use_container_width=True, caption="Predicted vs Actual Flight Prices")
    else:
        st.warning("Model performance comparison image not found.")

with tab4:
    residual_img = get_asset_path("residual_plot.png")
    if residual_img:
        st.image(residual_img, use_container_width=True, caption="Residuals vs Predicted Prices")
    else:
        st.warning("Residual plot diagnostics image not found.")
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
