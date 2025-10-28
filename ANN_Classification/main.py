# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
# import pandas as pd
# import pickle
#
# # Load the trained model
#
# model = tf.keras.models.load_model('model.h5')
#
# # load the encoders and scalers
#
# with open('onehot_encoder_geo.pkl', 'rb') as file:
#     onehot_encoder_geo = pickle.load(file)
#
# with open('label_encoder_gender.pkl', 'rb') as file:
#     label_encoder_gender = pickle.load(file)
#
# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)
#
# # streamlit app
#
# st.title('Customer Churn Prediction')
#
# # User input
#
# geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
# gender = st.selectbox('Gender', label_encoder_gender.classes_)
# age = st.slider('Age',18,92)
# balance = st.number_input('Balance')
# credit_score = st.number_input('Credit Score')
# estimated_salary = st.number_input('Estimated Salary')
# tenure = st.slider('Tenure', 0, 10)
# num_of_products = st.number_input('Number of Products', 1, 4)
# has_cr_card = st.selectbox('Has Credit Card', [0,1])
# is_active_member = st.selectbox('Is Active Member',[0,1])
#
# #prepare the input data
# input_data = pd.DataFrame({
#     'Credit_Card':[credit_score],
#     'Gender':[label_encoder_gender.transform([gender])[0]],
#     'Age':[age],
#     'Tenure':[tenure],
#     'Balance':[balance],
#     'NumOfProducts':[num_of_products],
#     'HasCrCard':[has_cr_card],
#     'IsActiveMember':[is_active_member],
#     'EstimatedSalary':[estimated_salary]
# }
# )
#
# # One-Hot encode "Geography"
#
# geo_encoded = onehot_encoder_geo.transorm([[geography]])
# geo_encoded_df = pd.DataFrame(geo_encoded.toarray(),columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
#
# # combining one-hot encoded columns with the original data
# input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
#
# #scale the input data
# input_data_scaled = scaler.transform(input_data)
#
# #Prediction Churn
# prediction = model.predict(input_data_scaled)
# prediction_prob = prediction[0][0]
#
# if prediction_prob > 0.5:
#     st.write("The customer is likely to churn.")
# else:
#     st.write("The customer is unlikely to churn.")
#


import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import warnings
import os

# --- Page Configuration ---
# Set the page title and icon
st.set_page_config(page_title="Churn Predictor", page_icon="⚡")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')
warnings.filterwarnings('ignore', category=FutureWarning, module='google.protobuf')


# --- Caching Functions ---
# Cache the loading of models and encoders to speed up the app
@st.cache_resource
def load_model_and_preprocessors():
    """Loads the trained model, scaler, and encoders."""
    try:
        model = tf.keras.models.load_model('model.h5')
    except Exception as e:
        st.error(f"Error loading model.h5: {e}")
        st.stop()

    try:
        with open('onehot_encoder_geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
    except FileNotFoundError:
        st.error("onehot_encoder_geo.pkl not found. Please train the model first.")
        st.stop()

    try:
        with open('label_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
    except FileNotFoundError:
        st.error("label_encoder_gender.pkl not found. Please train the model first.")
        st.stop()

    try:
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
    except FileNotFoundError:
        st.error("scaler.pkl not found. Please train the model first.")
        st.stop()

    return model, onehot_encoder_geo, label_encoder_gender, scaler


# Load all necessary files
model, onehot_encoder_geo, label_encoder_gender, scaler = load_model_and_preprocessors()

# --- Streamlit App UI ---
st.title('⚡ Customer Churn Prediction')
st.markdown("Enter the customer's details below to predict the likelihood of churn.")

# --- UI Layout ---
# Use columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    # BUG FIX #1: Your 'experiment.ipynb' uses 'CreditScore'.
    # Your old 'main.py' used 'Credit_Card', which would cause an error.
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])

with col2:
    # UI Improvement: Use st.radio for clearer 'Gender' selection
    gender = st.radio('Gender', label_encoder_gender.classes_, horizontal=True)
    age = st.slider('Age', 18, 92, 40)  # Add a default value

with col3:
    tenure = st.slider('Tenure (Years)', 0, 10, 5)  # Add a default value
    # BUG FIX #2: Your 'experiment.ipynb' uses 'NumOfProducts'.
    # Ensure the key in the final DataFrame matches this.
    num_of_products = st.number_input('Number of Products', 1, 4, 1)

# --- More Inputs ---
st.divider()

col4, col5 = st.columns(2)

with col4:
    balance = st.number_input('Balance', min_value=0.0, value=0.0, format="%.2f")
    # UI Improvement: Use st.radio for clearer Yes/No questions
    has_cr_card = st.radio('Has Credit Card?', ["Yes", "No"], horizontal=True)

with col5:
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0, format="%.2f")
    is_active_member = st.radio('Is Active Member?', ["Yes", "No"], horizontal=True)

# --- Prediction Logic ---
# UI Improvement: Use a button to trigger the prediction
if st.button('Predict Churn', type="primary"):

    # 1. Process inputs to match training data
    gender_encoded = label_encoder_gender.transform([gender])[0]
    has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
    is_active_member_encoded = 1 if is_active_member == "Yes" else 0

    # 2. Prepare the input DataFrame (with correct column names from training)
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],  # This must match the training column
        'HasCrCard': [has_cr_card_encoded],
        'IsActiveMember': [is_active_member_encoded],
        'EstimatedSalary': [estimated_salary]
    })

    # 3. One-Hot encode "Geography"
    # BUG FIX #3: Corrected typo 'transorm' -> 'transform'
    geo_encoded = onehot_encoder_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(
        geo_encoded.toarray(),
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # 4. Combine one-hot encoded columns with the rest of the data
    input_processed_df = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # 5. BUG FIX #4 (CRITICAL): Re-order columns to match the scaler's expectations
    # This is the most common source of errors.
    try:
        expected_features = scaler.feature_names_in_
        final_input_df = input_processed_df[expected_features]
    except KeyError as e:
        st.error(f"Error: A feature is missing or misnamed. {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error preparing features: {e}")
        st.stop()

    # 6. Scale the final input data
    input_data_scaled = scaler.transform(final_input_df)

    # 7. Make the Prediction
    prediction = model.predict(input_data_scaled)
    prediction_prob = prediction[0][0]

    # 8. Display the result
    # UI Improvement: Use st.metric and st.success/st.error for a clearer result
    st.subheader('Prediction Result')

    # Format probability as percentage
    prob_percentage = f"{prediction_prob:.2%}"

    if prediction_prob > 0.5:
        st.error(f"Customer is LIKELY to Churn (Probability: {prob_percentage})", icon="🚨")
    else:
        st.success(f"Customer is UNLIKELY to Churn (Probability: {prob_percentage})", icon="✅")

    st.markdown("---")
    st.subheader("Input Data Sent to Model (After Processing):")
    st.dataframe(final_input_df)
