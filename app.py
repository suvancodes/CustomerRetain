import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: transparent;
    }
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: bold;
        color: #667eea;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 16px;
        font-weight: 600;
        color: #555;
    }
    .prediction-box {
        padding: 40px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 20px 0;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3.5em;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
        font-weight: 700;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: white;
        font-size: 20px;
        font-weight: 300;
        margin-bottom: 30px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 18px;
        font-size: 20px;
        border-radius: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.6);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
    }
    .risk-badge {
        display: inline-block;
        padding: 10px 20px;
        border-radius: 50px;
        font-weight: bold;
        font-size: 18px;
        margin: 10px 0;
    }
    .high-risk {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
    }
    .medium-risk {
        background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%);
        color: white;
    }
    .low-risk {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e063 100%);
        color: white;
    }
    .action-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #667eea;
        color: #333;
    }
    .action-card h3 {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .action-card ul {
        color: #333;
    }
    .action-card ul li {
        margin: 10px 0;
        color: #333;
        line-height: 1.6;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #56ab2f 0%, #f7971e 50%, #ff416c 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('notebook/artifect/model.h5')

@st.cache_resource
def load_encoders():
    with open('notebook/artifect/level_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    with open('notebook/artifect/onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    
    with open('notebook/artifect/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return label_encoder_gender, onehot_encoder_geo, scaler

model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_encoders()

# Header
st.title('🎯 Customer Churn Predictor')
st.markdown("<p class='subtitle'>Predict customer churn probability using advanced machine learning</p>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar for input
with st.sidebar:
    st.header("📋 Customer Information")
    st.markdown("Please provide the following details:")
    
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 35)
    
    st.markdown("---")
    st.subheader("💰 Financial Details")
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=650)
    balance = st.number_input('💵 Balance', min_value=0.0, value=0.0, format="%.2f")
    estimated_salary = st.number_input('💼 Estimated Salary', min_value=0.0, value=50000.0, format="%.2f")
    
    st.markdown("---")
    st.subheader("🏦 Account Details")
    tenure = st.slider('📅 Tenure (Years)', 0, 10, 5)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 2)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox('✅ Is Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    predict_button = st.button('🔮 Predict Churn')

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if predict_button:
        with st.spinner('🔍 Analyzing customer data...'):
            # Prepare the input data
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary]
            })

            # One-hot encode 'Geography'
            geo_encoded = onehot_encoder_geo.transform([[geography]])
            geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

            # Combine one-hot encoded columns with input data
            input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Predict churn
            prediction = model.predict(input_data_scaled)
            prediction_proba = prediction[0][0]

            # Display results with styling
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            
            st.markdown(f"<h2 style='text-align: center; color: #667eea; font-weight: 700;'>📊 Prediction Results</h2>", unsafe_allow_html=True)
            
            # Risk Badge
            if prediction_proba >= 0.7:
                risk_level = "CRITICAL RISK"
                badge_class = "high-risk"
                risk_emoji = "🔴"
            elif prediction_proba >= 0.5:
                risk_level = "HIGH RISK"
                badge_class = "high-risk"
                risk_emoji = "🟠"
            elif prediction_proba >= 0.3:
                risk_level = "MEDIUM RISK"
                badge_class = "medium-risk"
                risk_emoji = "🟡"
            else:
                risk_level = "LOW RISK"
                badge_class = "low-risk"
                risk_emoji = "🟢"
            
            st.markdown(f"<center><div class='risk-badge {badge_class}'>{risk_emoji} {risk_level}</div></center>", unsafe_allow_html=True)
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric(label="🔻 Churn Probability", value=f"{prediction_proba:.1%}")
            with col_b:
                retention_prob = 1 - prediction_proba
                st.metric(label="✅ Retention Probability", value=f"{retention_prob:.1%}")
            with col_c:
                confidence = max(prediction_proba, retention_prob)
                st.metric(label="🎯 Confidence", value=f"{confidence:.1%}")
            
            # Progress bar
            st.markdown("<p style='text-align: center; font-weight: 600; margin-top: 20px;'>Churn Risk Level</p>", unsafe_allow_html=True)
            st.progress(float(prediction_proba))
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Dynamic recommendations based on percentage
            if prediction_proba >= 0.7:  # 70%+ Critical
                st.error(f"🚨 **CRITICAL ALERT**: Customer has {prediction_proba:.1%} probability of churning!")
                st.markdown("""
                <div class='action-card'>
                <h3>🎯 Immediate Actions Required:</h3>
                <ul>
                    <li>🔥 <b>URGENT:</b> Assign dedicated account manager within 24 hours</li>
                    <li>💰 Offer premium retention package (20-30% discount)</li>
                    <li>📞 Schedule executive-level call immediately</li>
                    <li>🎁 Provide exclusive loyalty benefits and rewards</li>
                    <li>⚡ Fast-track any pending issues or complaints</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
            elif prediction_proba >= 0.5:  # 50-70% High Risk
                st.warning(f"⚠️ **HIGH RISK**: Customer has {prediction_proba:.1%} probability of churning")
                st.markdown("""
                <div class='action-card'>
                <h3>🎯 Priority Actions:</h3>
                <ul>
                    <li>📞 Personal outreach call within 48 hours</li>
                    <li>💰 Offer 10-15% retention discount</li>
                    <li>🎁 Send special retention offer package</li>
                    <li>📊 Conduct satisfaction survey</li>
                    <li>🤝 Review and address service gaps</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
            elif prediction_proba >= 0.3:  # 30-50% Medium Risk
                st.info(f"📊 **MEDIUM RISK**: Customer has {prediction_proba:.1%} probability of churning")
                st.markdown("""
                <div class='action-card'>
                <h3>🎯 Recommended Actions:</h3>
                <ul>
                    <li>📧 Send personalized engagement email</li>
                    <li>🎁 Offer loyalty rewards</li>
                    <li>📊 Monitor account activity closely</li>
                    <li>🤝 Schedule quarterly check-in</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
            else:  # Below 30% - Low Risk
                st.success(f"✅ **LOW RISK**: Customer has only {prediction_proba:.1%} probability of churning")
                st.markdown("""
                <div class='action-card'>
                <h3>🎯 Maintenance Actions:</h3>
                <ul>
                    <li>📧 Continue regular engagement</li>
                    <li>🎁 Include in loyalty programs</li>
                    <li>📊 Standard monitoring</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Additional insights
            with st.expander("📊 View Detailed Customer Profile"):
                st.write(f"**Credit Score:** {credit_score}")
                st.write(f"**Age:** {age}")
                st.write(f"**Geography:** {geography}")
                st.write(f"**Gender:** {gender}")
                st.write(f"**Balance:** ${balance:,.2f}")
                st.write(f"**Tenure:** {tenure} years")
            
            # Risk factors analysis
            with st.expander("🔍 Risk Factors Analysis"):
                st.write("Key factors influencing this prediction:")
                if age > 50:
                    st.write("• Age above 50 may increase churn risk")
                if balance == 0:
                    st.write("• Zero balance indicates potential disengagement")
                if is_active_member == 0:
                    st.write("• Inactive member status increases risk")
                if num_of_products == 1:
                    st.write("• Single product customers are more likely to churn")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: white; font-size: 16px;'>Made with ❤️ using Streamlit & TensorFlow | © 2025 Customer Analytics</p>", unsafe_allow_html=True)
