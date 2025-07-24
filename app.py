import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import tempfile
import os
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="AI Waste Classification System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ComprehensiveWasteClassifier:
    def __init__(self, model_path):
        """Initialize the waste classification model"""
        try:
            self.model = YOLO(model_path)
            self.class_names = [
                'battery', 'cardboard', 'clothes', 'food-organics', 'glass', 
                'metal', 'miscellaneous-trash', 'mixed', 'paper', 'procelain', 
                'rubber', 'textile', 'vegetation', 'HDPE', 'LDPA', 'other', 
                'PC', 'PET', 'PP', 'PS', 'PVC', 'shoes'
            ]
            self.num_classes = len(self.class_names)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            self.model = None
    
    def predict_image(self, image):
        """Make prediction on a single image"""
        if self.model is None:
            return None
        
        try:
            # Convert PIL image to format suitable for YOLO
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Make prediction
            results = self.model.predict(image_array, verbose=False, conf=0.1)
            
            if results and len(results) > 0:
                probs = results[0].probs
                pred_idx = probs.top1
                confidence = probs.top1conf.item()
                all_probs = probs.data.cpu().numpy()
                
                # Calculate additional metrics
                prediction_certainty = float(np.max(all_probs) - np.sort(all_probs)[-2])
                shannon_entropy = float(-np.sum(all_probs * np.log(all_probs + 1e-15)))
                
                return {
                    'success': True,
                    'predicted_class': self.class_names[pred_idx],
                    'predicted_index': int(pred_idx),
                    'confidence': float(confidence),
                    'all_probabilities': all_probs,
                    'prediction_certainty': prediction_certainty,
                    'shannon_entropy': shannon_entropy,
                    'top_5_predictions': self._get_top_k_predictions(all_probs, k=5)
                }
            else:
                return {'success': False, 'error': 'No prediction results'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _get_top_k_predictions(self, probabilities, k=5):
        """Get top K predictions with class names and probabilities"""
        top_indices = np.argsort(probabilities)[-k:][::-1]
        return [(self.class_names[idx], float(probabilities[idx])) for idx in top_indices]

@st.cache_resource
def load_model(model_path):
    """Cache the model loading to avoid reloading"""
    return ComprehensiveWasteClassifier(model_path)

def capture_image_from_camera():
    """Capture image using camera input"""
    camera_input = st.camera_input("📷 Take a photo of waste item")
    
    if camera_input is not None:
        # Convert camera input to PIL Image
        image = Image.open(camera_input)
        return image
    return None

def upload_image_file():
    """Handle file upload"""
    uploaded_file = st.file_uploader(
        "📁 Upload an image file", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_file)
        return image
    return None

def display_prediction_results(result, image):
    """Display prediction results in a nice format"""
    if not result['success']:
        st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
        return
    
    # Main prediction display
    confidence = result['confidence']
    predicted_class = result['predicted_class']
    
    # Confidence color coding
    if confidence >= 0.8:
        conf_class = "confidence-high"
        conf_icon = "🟢"
    elif confidence >= 0.5:
        conf_class = "confidence-medium"
        conf_icon = "🟡"
    else:
        conf_class = "confidence-low"
        conf_icon = "🔴"
    
    # Display main prediction
    st.markdown(f"""
    <div class="prediction-box">
        <h2>🎯 Prediction Results</h2>
        <h1>{predicted_class.upper()}</h1>
        <p>Confidence: <span class="{conf_class}">{conf_icon} {confidence:.2%}</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for detailed results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Top 5 Predictions")
        
        # Create DataFrame for top predictions
        top_predictions = result['top_5_predictions']
        df_predictions = pd.DataFrame(top_predictions, columns=['Class', 'Probability'])
        df_predictions['Percentage'] = df_predictions['Probability'].apply(lambda x: f"{x:.2%}")
        
        # Display as table
        st.dataframe(
            df_predictions[['Class', 'Percentage']], 
            use_container_width=True,
            hide_index=True
        )
        
        # Create bar chart
        fig_bar = px.bar(
            df_predictions.head(5), 
            x='Probability', 
            y='Class',
            orientation='h',
            title="Top 5 Predictions",
            color='Probability',
            color_continuous_scale='viridis'
        )
        fig_bar.update_layout(height=300, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Detailed Analysis")
        
        # Metrics display
        metrics_data = {
            "Prediction Certainty": f"{result['prediction_certainty']:.4f}",
            "Shannon Entropy": f"{result['shannon_entropy']:.4f}",
            "Max Probability": f"{np.max(result['all_probabilities']):.4f}",
            "Min Probability": f"{np.min(result['all_probabilities']):.4f}",
            "Standard Deviation": f"{np.std(result['all_probabilities']):.4f}"
        }
        
        for metric, value in metrics_data.items():
            st.metric(label=metric, value=value)
        
        # All class probabilities pie chart
        st.subheader("🥧 All Class Probabilities")
        
        # Get top 10 for pie chart (to avoid clutter)
        all_probs = result['all_probabilities']
        class_names = [
            'battery', 'cardboard', 'clothes', 'food-organics', 'glass', 
            'metal', 'miscellaneous-trash', 'mixed', 'paper', 'procelain', 
            'rubber', 'textile', 'vegetation', 'HDPE', 'LDPA', 'other', 
            'PC', 'PET', 'PP', 'PS', 'PVC', 'shoes'
        ]
        
        # Create DataFrame for all probabilities
        df_all = pd.DataFrame({
            'Class': class_names,
            'Probability': all_probs
        }).sort_values('Probability', ascending=False)
        
        # Top 8 + Others
        top_8 = df_all.head(8)
        others_sum = df_all.tail(len(df_all)-8)['Probability'].sum()
        
        if others_sum > 0:
            others_row = pd.DataFrame({'Class': ['Others'], 'Probability': [others_sum]})
            pie_data = pd.concat([top_8, others_row], ignore_index=True)
        else:
            pie_data = top_8
        
        fig_pie = px.pie(
            pie_data, 
            values='Probability', 
            names='Class',
            title="Probability Distribution (Top Classes)"
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

def display_basic_model_info():
    """Display basic model information without JSON files"""
    st.subheader("🤖 Model Information")
    
    # Model specifications (hardcoded since we don't have JSON files)
    st.subheader("⚙️ Model Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Model Type**: YOLOv8n Classification")
        st.info("**Number of Classes**: 22")
        st.info("**Input Size**: 224x224")
        st.info("**Framework**: PyTorch + Ultralytics")
    
    with col2:
        st.info("**Task**: Multi-class Classification")
        st.info("**Categories**: Waste Types")
        st.info("**Output**: Class + Confidence Score")
        st.info("**Real-time**: Yes")
    
    # Class information
    st.subheader("🗂️ Waste Categories (22 Classes)")
    
    classes = [
        'battery', 'cardboard', 'clothes', 'food-organics', 'glass', 
        'metal', 'miscellaneous-trash', 'mixed', 'paper', 'procelain', 
        'rubber', 'textile', 'vegetation', 'HDPE', 'LDPA', 'other', 
        'PC', 'PET', 'PP', 'PS', 'PVC', 'shoes'
    ]
    
    # Display classes in a nice grid
    cols = st.columns(4)
    for i, class_name in enumerate(classes):
        col_idx = i % 4
        cols[col_idx].write(f"• {class_name}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">♻️ AI-Powered Waste Classification System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Advanced 22-class waste classification system
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Settings")
        
        # Model file selection
        model_path = st.text_input(
            "Model Path", 
            value="best.pt", 
            help="Path to your trained YOLO model file"
        )
        
        # Check if model file exists
        if not os.path.exists(model_path):
            st.error(f"❌ Model file '{model_path}' not found!")
            st.info("Please upload your 'best.pt' file to the same directory as this script.")
            st.stop()
        
        st.success(f"✅ Model file found: {model_path}")
        
        # Input method selection
        st.header("📥 Input Method")
        input_method = st.radio(
            "Choose input method:",
            ["📷 Camera", "📁 File Upload"],
            help="Select how you want to provide the image"
        )
        
        # Display model info toggle
        show_model_info = st.checkbox("Show Model Information", value=False)
    
    # Load model
    try:
        classifier = load_model(model_path)
        
        if classifier.model is None:
            st.error("Failed to load model. Please check the model file.")
            st.stop()
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Main content
    image = None
    
    # Input method handling
    if input_method == "📷 Camera":
        st.subheader("📷 Camera Input")
        st.info("Click the camera button below to take a photo of the waste item.")
        image = capture_image_from_camera()
        
    else:  # File Upload
        st.subheader("📁 File Upload")
        st.info("Upload an image file of the waste item you want to classify.")
        image = upload_image_file()
    
    # Process image if available
    if image is not None:
        # Display the image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Input Image", use_column_width=True)
        
        # Add classify button
        if st.button("🔍 Classify Waste", type="primary", use_container_width=True):
            with st.spinner("🤔 Analyzing image..."):
                # Make prediction
                result = classifier.predict_image(image)
                
                # Display results
                display_prediction_results(result, image)
    
    # Model information section
    if show_model_info:
        st.markdown("---")
        display_basic_model_info()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🌱 Comprehensive Waste Classification System | Built with Streamlit & YOLOv8 🌱</p>
        <p>22-class waste classification for smart recycling</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
