import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import warnings
import os

# --- Page Configuration ---
st.set_page_config(page_title="Salary Predictor", page_icon="💰")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', category=FutureWarning, module='google.protobuf')


# --- Caching Functions ---
# Cache the loading of models and encoders to speed up the app
@st.cache_resource
def load_regression_model_and_preprocessors():
    """Loads the trained regression model, scaler, and encoders."""
    try:
        # Load the regression model
        model = tf.keras.models.load_model('regression_model.h5')
    except Exception as e:
        st.error(f"Error loading regression_model.h5: {e}")
        st.error("Please ensure the regression model was trained and saved correctly.")
        st.stop()

    # Load the same preprocessors used during training
    try:
        with open('onehot_encoder_geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
    except FileNotFoundError:
        st.error("onehot_encoder_geo.pkl not found. It should be created by the training notebook.")
        st.stop()

    try:
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
    except FileNotFoundError:
        st.error("label_encoder_gender.pkl not found. It should be created by the training notebook.")
        st.stop()

    try:
        # **Important:** Ensure this scaler was fitted on the features *without* EstimatedSalary
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
    except FileNotFoundError:
        st.error("scaler.pkl not found. It should be created by the training notebook.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading scaler.pkl: {e}")
        st.stop()

    return model, onehot_encoder_geo, label_encoder_gender, scaler


# Load all necessary files
model, onehot_encoder_geo, label_encoder_gender, scaler = load_regression_model_and_preprocessors()

# --- Streamlit App UI ---
st.title('💰 Customer Estimated Salary Prediction')
st.markdown("Enter the customer's details below to predict their estimated salary.")

# --- UI Layout ---
# Use columns for better organization
col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    # The 'Exited' status is needed as an input feature based on the notebook
    exited_status = st.radio('Customer has Exited?', ["Yes", "No"], horizontal=True, index=1)

with col2:
    gender = st.radio('Gender', label_encoder_gender.classes_, horizontal=True)
    age = st.slider('Age', 18, 92, 40)
    tenure = st.slider('Tenure (Years)', 0, 10, 5)

with col3:
    num_of_products = st.number_input('Number of Products', 1, 4, 1)
    has_cr_card = st.radio('Has Credit Card?', ["Yes", "No"], horizontal=True)
    is_active_member = st.radio('Is Active Member?', ["Yes", "No"], horizontal=True)

# Balance input spans full width
balance = st.number_input('Balance', min_value=0.0, value=0.0, format="%.2f")

st.divider()

# --- Prediction Logic ---
if st.button('Predict Salary', type="primary"):

    # 1. Process inputs
    gender_encoded = label_encoder_gender.transform([gender])[0]
    has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
    is_active_member_encoded = 1 if is_active_member == "Yes" else 0
    exited_encoded = 1 if exited_status == "Yes" else 0  # Exited is a feature here

    # 2. Prepare the input DataFrame - must match columns used for training X
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card_encoded],
        'IsActiveMember': [is_active_member_encoded],
        'Exited': [exited_encoded]  # Include Exited status
        # 'EstimatedSalary' is NOT included here, it's what we predict
    })

    # 3. One-Hot encode "Geography"
    geo_encoded = onehot_encoder_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded.toarray(),
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # 4. Combine encoded Geography with the rest
    input_processed_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # 5. CRITICAL: Re-order columns to match the scaler's training order
    try:
        # Check if the scaler has feature names (it should)
        if not hasattr(scaler, 'feature_names_in_'):
            st.error("Scaler object was not fitted correctly or doesn't have feature names stored. Cannot proceed.")
            st.stop()

        expected_features = scaler.feature_names_in_

        # Verify all expected columns are present
        missing_cols = set(expected_features) - set(input_processed_df.columns)
        if missing_cols:
            st.error(f"Input data is missing the following columns expected by the scaler: {missing_cols}")
            st.stop()

        final_input_df = input_processed_df[expected_features]

    except KeyError as e:
        st.error(f"Error aligning columns: Column '{e}' not found in the processed input.")
        st.error(f"Expected columns: {list(scaler.feature_names_in_)}")
        st.error(f"Input columns: {list(input_processed_df.columns)}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while preparing features: {e}")
        st.stop()

    # 6. Scale the final input data
    try:
        input_data_scaled = scaler.transform(final_input_df)
    except ValueError as e:
        st.error(f"Error during scaling: {e}")
        st.error("This usually means the number or order of features is incorrect.")
        st.dataframe(final_input_df)  # Show the data being sent to scaler
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during scaling: {e}")
        st.stop()

    # 7. Make the Prediction
    predicted_salary = model.predict(input_data_scaled)
    predicted_value = predicted_salary[0][0]  # Get the scalar value

    # 8. Display the result
    st.subheader('Prediction Result')
    st.metric(label="Predicted Estimated Salary", value=f"${predicted_value:,.2f}")  # Format as currency

    st.markdown("---")
    st.subheader("Input Data Sent to Model (After Processing & Scaling):")
    st.dataframe(final_input_df)  # Show the final pre-scaled data for verification