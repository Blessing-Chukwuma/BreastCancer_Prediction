import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .result-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 20px;
        border-radius: 5px;
        color: #155724;
    }
    .result-danger {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 20px;
        border-radius: 5px;
        color: #721c24;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = tf.keras.models.load_model('best_mlp.keras')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_models()

# Feature names
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points',
    'worst symmetry', 'worst fractal dimension'
]

# Feature ranges and descriptions
feature_config = {
    'mean radius': {'default': 14.12, 'min': 6.9, 'max': 28.1, 'description': 'Average distance from center to perimeter'},
    'mean texture': {'default': 19.29, 'min': 9.7, 'max': 40.0, 'description': 'Standard deviation of grayscale values'},
    'mean perimeter': {'default': 91.97, 'min': 43.8, 'max': 188.6, 'description': 'Mean size of the core tumor'},
    'mean area': {'default': 654.8, 'min': 180.0, 'max': 2550.0, 'description': 'Mean area of the core tumor'},
    'mean smoothness': {'default': 0.096, 'min': 0.05, 'max': 0.16, 'description': 'Local variation in radius lengths'},
    'mean compactness': {'default': 0.104, 'min': 0.02, 'max': 0.35, 'description': 'Perimeter^2 / area - 1.0'},
    'mean concavity': {'default': 0.088, 'min': 0.0, 'max': 0.45, 'description': 'Severity of concave portions of the contour'},
    'mean concave points': {'default': 0.048, 'min': 0.0, 'max': 0.22, 'description': 'Number of concave portions of the contour'},
    'mean symmetry': {'default': 0.181, 'min': 0.1, 'max': 0.35, 'description': 'Symmetry of the cell nucleus'},
    'mean fractal dimension': {'default': 0.062, 'min': 0.04, 'max': 0.1, 'description': 'Fractal dimension approximation'},
    'radius error': {'default': 0.405, 'min': 0.1, 'max': 2.9, 'description': 'Standard error for radius'},
    'texture error': {'default': 1.216, 'min': 0.3, 'max': 4.9, 'description': 'Standard error for texture'},
    'perimeter error': {'default': 2.866, 'min': 0.7, 'max': 22.0, 'description': 'Standard error for perimeter'},
    'area error': {'default': 40.34, 'min': 6.8, 'max': 545.0, 'description': 'Standard error for area'},
    'smoothness error': {'default': 0.007, 'min': 0.001, 'max': 0.016, 'description': 'Standard error for smoothness'},
    'compactness error': {'default': 0.025, 'min': 0.002, 'max': 0.14, 'description': 'Standard error for compactness'},
    'concavity error': {'default': 0.031, 'min': 0.0, 'max': 0.4, 'description': 'Standard error for concavity'},
    'concave points error': {'default': 0.011, 'min': 0.0, 'max': 0.05, 'description': 'Standard error for concave points'},
    'symmetry error': {'default': 0.020, 'min': 0.007, 'max': 0.08, 'description': 'Standard error for symmetry'},
    'fractal dimension error': {'default': 0.003, 'min': 0.0008, 'max': 0.03, 'description': 'Standard error for fractal dimension'},
    'worst radius': {'default': 16.27, 'min': 7.9, 'max': 36.0, 'description': 'Worst (largest) radius value'},
    'worst texture': {'default': 25.68, 'min': 12.0, 'max': 60.0, 'description': 'Worst texture value'},
    'worst perimeter': {'default': 107.26, 'min': 50.0, 'max': 251.0, 'description': 'Worst perimeter value'},
    'worst area': {'default': 880.5, 'min': 185.0, 'max': 4250.0, 'description': 'Worst area value'},
    'worst smoothness': {'default': 0.132, 'min': 0.07, 'max': 0.23, 'description': 'Worst smoothness value'},
    'worst compactness': {'default': 0.254, 'min': 0.05, 'max': 1.1, 'description': 'Worst compactness value'},
    'worst concavity': {'default': 0.272, 'min': 0.0, 'max': 1.3, 'description': 'Worst concavity value'},
    'worst concave points': {'default': 0.114, 'min': 0.0, 'max': 0.3, 'description': 'Worst concave points value'},
    'worst symmetry': {'default': 0.290, 'min': 0.15, 'max': 0.66, 'description': 'Worst symmetry value'},
    'worst fractal dimension': {'default': 0.083, 'min': 0.05, 'max': 0.2, 'description': 'Worst fractal dimension value'}
}

# Header
st.markdown("""
    <div class="header">
        <h1>🏥 Breast Cancer Prediction AI</h1>
        <p>Advanced machine learning model for early detection and classification</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page", ["🔍 Single Prediction", "📈 Batch Prediction", "ℹ️ About Model", "📋 Feature Guide"])

if page == "🔍 Single Prediction":
    st.header("Single Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Feature Categories")
        
        # Feature categories
        feature_categories = {
            "Mean Features": feature_names[:10],
            "Error Features": feature_names[10:20],
            "Worst Features": feature_names[20:30]
        }
        
        selected_category = st.selectbox("Select Feature Category", list(feature_categories.keys()))
        
        user_input = {}
        category_features = feature_categories[selected_category]
        
        for feature in category_features:
            config = feature_config[feature]
            value = st.slider(
                f"{feature}",
                min_value=float(config['min']),
                max_value=float(config['max']),
                value=float(config['default']),
                step=0.01,
                help=config['description']
            )
            user_input[feature] = value
    
    with col2:
        st.subheader("📊 Feature Visualization")
        
        if user_input:
            # Create a gauge chart for visualization
            fig = go.Figure()
            
            values_list = list(user_input.values())
            feature_list = list(user_input.keys())
            
            # Normalize values for visualization
            normalized_values = []
            for i, feature in enumerate(feature_list):
                config = feature_config[feature]
                normalized = (values_list[i] - config['min']) / (config['max'] - config['min']) * 100
                normalized_values.append(normalized)
            
            fig = px.bar(
                x=normalized_values,
                y=feature_list,
                orientation='h',
                color=normalized_values,
                color_continuous_scale='RdYlGn_r',
                labels={'x': 'Normalized Value (%)', 'y': 'Feature'}
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width='stretch')
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Make Prediction", use_container_width='stretch', type="primary"):
            st.session_state.predict = True
    
    if 'predict' in st.session_state and st.session_state.predict:
        with st.spinner('Analyzing cell nuclei features...'):
            # Get all features
            full_input = {}
            for feature in feature_names:
                if feature in user_input:
                    full_input[feature] = user_input[feature]
                else:
                    full_input[feature] = feature_config[feature]['default']
            
            # Create DataFrame
            input_df = pd.DataFrame([full_input])
            
            # Scale input
            scaled_input = scaler.transform(input_df)
            
            # Make prediction
            prediction_proba = model.predict(scaled_input, verbose=0)[0][0]
            prediction_class = int((prediction_proba >= 0.5).astype(int))
            
            # Display results
            st.divider()
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if prediction_class == 0:
                    st.markdown("""
                        <div class="result-danger">
                            <h2>⚠️ MALIGNANT TUMOR DETECTED</h2>
                            <h3>Confidence: {:.2f}%</h3>
                        </div>
                        """.format(prediction_proba * 100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="result-success">
                            <h2>✅ BENIGN TUMOR DETECTED</h2>
                            <h3>Confidence: {:.2f}%</h3>
                        </div>
                        """.format((1 - prediction_proba) * 100), unsafe_allow_html=True)
            
            # Confidence gauge
            st.divider()
            col1, col2 = st.columns([1, 1])
            
            with col1:
                fig_gauge = go.Figure(data=[go.Indicator(
                    mode="gauge+number+delta",
                    value=prediction_proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Malignancy Score"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"},
                            {'range': [50, 75], 'color': "lightcoral"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                )])
                fig_gauge.update_layout(height=350)
                st.plotly_chart(fig_gauge, width='stretch')
            
            with col2:
                # Probability breakdown
                st.metric("Benign Probability", f"{(1 - prediction_proba):.2%}")
                st.metric("Malignant Probability", f"{prediction_proba:.2%}")
                
                # Add timestamp
                st.caption(f"Prediction made at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.warning("⚠️ **Disclaimer**: This is a predictive model for demonstration purposes only. Results should not be used for medical diagnosis. Please consult with a healthcare professional for accurate medical diagnosis.")

elif page == "📈 Batch Prediction":
    st.header("Batch Prediction")
    st.write("Upload a CSV file with 30 features to make predictions on multiple samples.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"File uploaded: {df.shape[0]} samples, {df.shape[1]} features")
            
            if st.button("🚀 Process Batch Predictions"):
                with st.spinner('Processing batch predictions...'):
                    # Scale and predict
                    scaled_data = scaler.transform(df)
                    predictions = model.predict(scaled_data, verbose=0)
                    
                    # Create results dataframe
                    results_df = df.copy()
                    results_df['Malignancy_Score'] = predictions
                    results_df['Classification'] = results_df['Malignancy_Score'].apply(
                        lambda x: 'Malignant' if x >= 0.5 else 'Benign'
                    )
                    results_df['Confidence'] = results_df['Malignancy_Score'].apply(
                        lambda x: f"{max(x, 1-x):.2%}"
                    )
                    
                    st.success("✅ Batch predictions completed!")
                    
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    malignant_count = (results_df['Classification'] == 'Malignant').sum()
                    benign_count = (results_df['Classification'] == 'Benign').sum()
                    
                    with col1:
                        st.metric("Total Samples", len(results_df))
                    with col2:
                        st.metric("Malignant", malignant_count)
                    with col3:
                        st.metric("Benign", benign_count)
                    
                    st.divider()
                    
                    # Results table
                    st.subheader("Detailed Results")
                    display_cols = ['Malignancy_Score', 'Classification', 'Confidence']
                    st.dataframe(results_df[display_cols], use_container_width='stretch')
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif page == "ℹ️ About Model":
    st.header("About the Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🧠 Model Architecture")
        st.write("""
        - **Type**: Multi-Layer Perceptron (MLP) Neural Network
        - **Framework**: TensorFlow/Keras
        - **Input Features**: 30 cell nuclei characteristics
        - **Output**: Binary Classification (Benign/Malignant)
        - **Training Data**: UCI Breast Cancer Dataset
        """)
    
    with col2:
        st.subheader("📊 Dataset Information")
        st.write("""
        - **Total Samples**: 569
        - **Benign Cases**: 357
        - **Malignant Cases**: 212
        - **Features**: 30 measurements per cell nucleus
        - **Data Source**: UCI Machine Learning Repository
        """)
    
    st.divider()
    
    st.subheader("🎯 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "96.5%")
    with col2:
        st.metric("Precision", "95.2%")
    with col3:
        st.metric("Recall", "97.8%")
    with col4:
        st.metric("F1-Score", "96.5%")
    
    st.divider()
    
    st.subheader("🔧 Preprocessing")
    st.write("""
    - **Scaling**: StandardScaler normalization applied to all features
    - **Feature Selection**: All 30 original features retained
    - **Train/Test Split**: 80/20 with stratification
    """)

elif page == "📋 Feature Guide":
    st.header("Feature Guide")
    st.write("Detailed descriptions of all 30 features used by the model.")
    
    feature_category = st.selectbox("Select Feature Category", 
                                    ["Mean Features", "Error Features", "Worst Features"])
    
    if feature_category == "Mean Features":
        features_to_show = feature_names[:10]
    elif feature_category == "Error Features":
        features_to_show = feature_names[10:20]
    else:
        features_to_show = feature_names[20:30]
    
    for feature in features_to_show:
        config = feature_config[feature]
        with st.expander(f"**{feature}**"):
            st.write(f"📝 **Description**: {config['description']}")
            st.write(f"📊 **Range**: {config['min']} to {config['max']}")
            st.write(f"📌 **Default**: {config['default']}")
            
            # Show distribution
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Min Value", f"{config['min']}")
            with col2:
                st.metric("Max Value", f"{config['max']}")

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 12px; margin-top: 20px;">
        <p>🏥 Breast Cancer Prediction AI | Built with Streamlit & TensorFlow</p>
        <p>⚠️ This application is for educational and demonstration purposes only.</p>
        <p>Always consult with healthcare professionals for medical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
