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
    .welcome-card {
        padding: 40px;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.95);
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 20px 0;
        text-align: center;
    }
    .welcome-card h2 {
        color: #667eea;
        margin-bottom: 20px;
    }
    .welcome-card p {
        color: #555;
        font-size: 16px;
        line-height: 1.8;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('notebook/artifect/model.h5')
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_resource
def load_encoders():
    try:
        with open('notebook/artifect/level_encoder_gender.pkl', 'rb') as file:
            label_encoder_gender = pickle.load(file)
        
        with open('notebook/artifect/onehot_encoder_geo.pkl', 'rb') as file:
            onehot_encoder_geo = pickle.load(file)
        
        with open('notebook/artifect/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        return label_encoder_gender, onehot_encoder_geo, scaler
    except Exception as e:
        st.error(f"Failed to load encoders: {e}")
        return None, None, None

model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_encoders()

if model is None or label_encoder_gender is None:
    st.error("❌ Failed to load required files. Please check that the following files exist:")
    st.code("""
notebook/artifect/model.h5
notebook/artifect/level_encoder_gender.pkl
notebook/artifect/onehot_encoder_geo.pkl
notebook/artifect/scaler.pkl
    """)
    st.stop()

# Header
st.title('🎯 Customer Churn Predictor')
st.markdown("<p class='subtitle'>Predict customer churn probability using advanced Deep learning(ANN)</p>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar for input
with st.sidebar:
    st.header("📋 Customer Information")
    st.markdown("Please provide the following details:")
    
    # Create options with placeholder
    geo_options = ['-- Select Geography --'] + list(onehot_encoder_geo.categories_[0])
    gender_options = ['-- Select Gender --'] + list(label_encoder_gender.classes_)
    
    geography = st.selectbox('🌍 Geography', geo_options)
    gender = st.selectbox('👤 Gender', gender_options)
    age = st.number_input('🎂 Age', min_value=18, max_value=92, value=None, placeholder="Enter age")
    
    st.markdown("---")
    st.subheader("💰 Financial Details")
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=None, placeholder="Enter credit score")
    balance = st.number_input('💵 Balance', min_value=0.0, value=None, placeholder="Enter balance", format="%.2f")
    estimated_salary = st.number_input('💼 Estimated Salary', min_value=0.0, value=None, placeholder="Enter salary", format="%.2f")
    
    st.markdown("---")
    st.subheader("🏦 Account Details")
    tenure = st.number_input('📅 Tenure (Years)', min_value=0, max_value=10, value=None, placeholder="Enter years")
    num_of_products = st.number_input('📦 Number of Products', min_value=1, max_value=4, value=None, placeholder="Enter number")
    has_cr_card = st.selectbox('💳 Has Credit Card', ['-- Select --', 'No', 'Yes'])
    is_active_member = st.selectbox('✅ Is Active Member', ['-- Select --', 'No', 'Yes'])
    
    predict_button = st.button('🔮 Predict Churn')

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if predict_button:
        # Validate all inputs
        validation_errors = []
        
        if geography == '-- Select Geography --':
            validation_errors.append("Geography")
        if gender == '-- Select Gender --':
            validation_errors.append("Gender")
        if age is None:
            validation_errors.append("Age")
        if credit_score is None:
            validation_errors.append("Credit Score")
        if balance is None:
            validation_errors.append("Balance")
        if estimated_salary is None:
            validation_errors.append("Estimated Salary")
        if tenure is None:
            validation_errors.append("Tenure")
        if num_of_products is None:
            validation_errors.append("Number of Products")
        if has_cr_card == '-- Select --':
            validation_errors.append("Has Credit Card")
        if is_active_member == '-- Select --':
            validation_errors.append("Is Active Member")
        
        if validation_errors:
            st.error(f"⚠️ Please fill in the following fields: **{', '.join(validation_errors)}**")
        else:
            # Convert Yes/No to 1/0
            has_cr_card_val = 1 if has_cr_card == 'Yes' else 0
            is_active_val = 1 if is_active_member == 'Yes' else 0
            
            with st.spinner('🔍 Analyzing customer data...'):
                # Prepare the input data
                input_data = pd.DataFrame({
                    'CreditScore': [credit_score],
                    'Gender': [label_encoder_gender.transform([gender])[0]],
                    'Age': [age],
                    'Tenure': [tenure],
                    'Balance': [balance],
                    'NumOfProducts': [num_of_products],
                    'HasCrCard': [has_cr_card_val],
                    'IsActiveMember': [is_active_val],
                    'EstimatedSalary': [estimated_salary]
                })

                # One-hot encode 'Geography'
                geo_encoded = onehot_encoder_geo.transform([[geography]])
                geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

                # Combine one-hot encoded columns with input data
                input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
                
                # Scale the input data
                input_data_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_data_scaled)
                prediction_proba = prediction[0][0]
                
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                # Determine risk level and display badge
                if prediction_proba >= 0.5:
                    st.markdown("<span class='risk-badge high-risk'>🚨 HIGH RISK</span>", unsafe_allow_html=True)
                elif prediction_proba >= 0.3:
                    st.markdown("<span class='risk-badge medium-risk'>⚠️ MEDIUM RISK</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='risk-badge low-risk'>✅ LOW RISK</span>", unsafe_allow_html=True)
                
                # Display metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Churn Probability", f"{prediction_proba:.1%}")
                with col_b:
                    st.metric("Retention Probability", f"{1-prediction_proba:.1%}")
                with col_c:
                    st.metric("Confidence", f"{max(prediction_proba, 1-prediction_proba):.1%}")
                
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
                
                # Additional insights
                with st.expander("📊 View Detailed Customer Profile"):
                    profile_col1, profile_col2 = st.columns(2)
                    with profile_col1:
                        st.write(f"**Credit Score:** {credit_score}")
                        st.write(f"**Age:** {age}")
                        st.write(f"**Geography:** {geography}")
                        st.write(f"**Gender:** {gender}")
                        st.write(f"**Balance:** ${balance:,.2f}")
                    with profile_col2:
                        st.write(f"**Tenure:** {tenure} years")
                        st.write(f"**Number of Products:** {num_of_products}")
                        st.write(f"**Has Credit Card:** {has_cr_card}")
                        st.write(f"**Active Member:** {is_active_member}")
                        st.write(f"**Estimated Salary:** ${estimated_salary:,.2f}")
                
                # Risk factors analysis
                with st.expander("🔍 Risk Factors Analysis"):
                    st.write("**Key factors influencing this prediction:**")
                    risk_factors = []
                    positive_factors = []
                    
                    if age > 50:
                        risk_factors.append("• Age above 50 may increase churn risk")
                    if balance == 0:
                        risk_factors.append("• Zero balance indicates potential disengagement")
                    if is_active_val == 0:
                        risk_factors.append("• Inactive member status increases risk")
                    if num_of_products == 1:
                        risk_factors.append("• Single product customers are more likely to churn")
                    if credit_score < 500:
                        risk_factors.append("• Low credit score may indicate financial stress")
                    
                    if tenure >= 5:
                        positive_factors.append("• Long tenure indicates customer loyalty")
                    if num_of_products >= 3:
                        positive_factors.append("• Multiple products increase engagement")
                    if is_active_val == 1:
                        positive_factors.append("• Active member status is positive")
                    if credit_score >= 700:
                        positive_factors.append("• Good credit score indicates stability")
                    
                    if risk_factors:
                        st.markdown("**⚠️ Risk Factors:**")
                        for factor in risk_factors:
                            st.write(factor)
                    
                    if positive_factors:
                        st.markdown("**✅ Positive Factors:**")
                        for factor in positive_factors:
                            st.write(factor)
                    
                    if not risk_factors and not positive_factors:
                        st.write("No significant risk or positive factors identified.")
    
    else:
        # Welcome message when no prediction is made
        st.markdown("""
        <div class='welcome-card'>
            <h2>👋 Welcome!</h2>
            <p>
                Use the sidebar on the left to enter customer information.<br><br>
                Fill in all the required fields and click <b>"🔮 Predict Churn"</b> 
                to analyze the customer's churn probability.<br><br>
                The prediction model will provide:
            </p>
            <ul style='text-align: left; color: #555;'>
                <li>📊 Churn probability percentage</li>
                <li>🎯 Risk level assessment</li>
                <li>💡 Recommended retention actions</li>
                <li>🔍 Detailed risk factor analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: white; font-size: 16px;'>Made with ❤️ using Streamlit & TensorFlow | © 2025 Customer Analytics</p>", unsafe_allow_html=True)
