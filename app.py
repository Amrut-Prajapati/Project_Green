import streamlit as st
import os
import sys
import warnings
import tempfile
import requests
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Waste Classification App",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1edff;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class WasteClassificationApp:
    def __init__(self):
        self.model = None
        self.classes = [
            'HDPE', 'LDPA', 'PC', 'PET', 'PP', 'PS', 'PVC',
            'cardboard', 'glass', 'metal', 'paper', 'plastic',
            'trash', 'aluminium', 'biological', 'brown-glass',
            'clothes', 'green-glass', 'shoes', 'white-glass',
            'battery', 'e-waste'
        ]
        self.github_model_url = None  # Will be set from user input or config
        self.load_model()
    
    def download_model_from_github(self, github_url, local_path="best.pt"):
        """Download model from GitHub raw URL"""
        try:
            st.info(f"📥 Downloading model from GitHub...")
            response = requests.get(github_url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            st.success(f"✅ Model downloaded successfully to {local_path}")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return False
    
    @st.cache_resource
    def load_model(_self, github_url=None):
        """Load the trained model from various sources"""
        try:
            # Priority order for loading model
            model_paths = [
                "best.pt",
                "models/best.pt",
                "weights/best.pt",
                "./best.pt"
            ]
            
            # First, try to load from local paths
            for path in model_paths:
                if os.path.exists(path):
                    st.success(f"✅ Loading model from: {path}")
                    model = YOLO(path)
                    return model
            
            # If no local model found and GitHub URL provided, download it
            if github_url:
                if _self.download_model_from_github(github_url, "best.pt"):
                    model = YOLO("best.pt")
                    return model
            
            # Fallback to pretrained model
            st.warning("⚠️ Local trained model not found. Using pretrained YOLOv8n-cls model.")
            st.info("💡 To use your custom model, provide the GitHub raw URL in the sidebar.")
            model = YOLO('yolov8n-cls.pt')
            return model
            
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None
    
    def predict_image(self, image):
        """Make prediction on uploaded image"""
        if self.model is None:
            return None, None
        
        try:
            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                
                # Make prediction
                results = self.model.predict(tmp_file.name, verbose=False)
                
                if results and hasattr(results[0], 'probs'):
                    result = results[0]
                    
                    # Get top prediction
                    if hasattr(result.probs, 'top1') and hasattr(result.probs, 'top1conf'):
                        pred_class_idx = int(result.probs.top1)
                        confidence = float(result.probs.top1conf)
                        
                        # Map to class name if within range
                        if pred_class_idx < len(self.classes):
                            pred_class = self.classes[pred_class_idx]
                        else:
                            pred_class = f"Class_{pred_class_idx}"
                        
                        return pred_class, confidence
                    else:
                        # Fallback: get highest probability
                        probs = result.probs.data.cpu().numpy()
                        pred_class_idx = np.argmax(probs)
                        confidence = float(probs[pred_class_idx])
                        
                        if pred_class_idx < len(self.classes):
                            pred_class = self.classes[pred_class_idx]
                        else:
                            pred_class = f"Class_{pred_class_idx}"
                        
                        return pred_class, confidence
                
                # Clean up
                os.unlink(tmp_file.name)
                
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return None, None
        
        return None, None
    
    def get_top_predictions(self, image, top_k=5):
        """Get top K predictions with confidence scores"""
        if self.model is None:
            return []
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                results = self.model.predict(tmp_file.name, verbose=False)
                
                if results and hasattr(results[0], 'probs'):
                    probs = results[0].probs.data.cpu().numpy()
                    
                    # Get top K predictions
                    top_indices = np.argsort(probs)[::-1][:top_k]
                    top_predictions = []
                    
                    for idx in top_indices:
                        if idx < len(self.classes):
                            class_name = self.classes[idx]
                        else:
                            class_name = f"Class_{idx}"
                        
                        confidence = float(probs[idx])
                        top_predictions.append((class_name, confidence))
                    
                    return top_predictions
                
                os.unlink(tmp_file.name)
                
        except Exception as e:
            st.error(f"❌ Error getting top predictions: {e}")
            return []
        
        return []

def main():
    # Header
    st.markdown('<h1 class="main-header">♻️ Waste Classification System</h1>', unsafe_allow_html=True)
    
    # Sidebar for GitHub URL input
    st.sidebar.title("⚙️ Configuration")
    
    # GitHub model URL input
    github_model_url = st.sidebar.text_input(
        "🔗 GitHub Model URL (Raw)",
        placeholder="https://github.com/username/repo/raw/main/best.pt",
        help="Paste the raw GitHub URL of your best.pt file"
    )
    
    # Initialize app
    app = WasteClassificationApp()
    
    # Load model with GitHub URL if provided
    if github_model_url and github_model_url.strip():
        if st.sidebar.button("📥 Download Model from GitHub"):
            app.model = app.load_model(github_model_url.strip())
    
    # Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📤 Upload & Predict", "📊 Model Performance", "ℹ️ About", "🚀 Setup Guide"]
    )
    
    if page == "🏠 Home":
        show_home_page(app)
    elif page == "📤 Upload & Predict":
        show_prediction_page(app)
    elif page == "📊 Model Performance":
        show_performance_page(app)
    elif page == "ℹ️ About":
        show_about_page()
    elif page == "🚀 Setup Guide":
        show_setup_guide()

def show_home_page(app):
    """Display home page"""
    st.markdown("## Welcome to the Waste Classification System")
    
    # Model status
    if app.model is not None:
        st.markdown("""
        <div class="success-box">
            <strong>✅ Model Status:</strong> Loaded and ready for predictions!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Model Status:</strong> No custom model loaded. Using default pretrained model.
            <br>To use your trained model, provide the GitHub URL in the sidebar.
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Purpose</h3>
            <p>Automatically classify different types of waste materials using AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Technology</h3>
            <p>YOLOv8 Classification Model with 22 waste categories</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>♻️ Impact</h3>
            <p>Helps improve recycling efficiency and waste management</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Supported classes
    st.markdown("### 🗂️ Supported Waste Categories")
    
    categories = {
        "Plastics": ["HDPE", "LDPA", "PC", "PET", "PP", "PS", "PVC", "plastic"],
        "Paper & Cardboard": ["cardboard", "paper"],
        "Glass": ["glass", "brown-glass", "green-glass", "white-glass"],
        "Metals": ["metal", "aluminium"],
        "Electronics": ["e-waste", "battery"],
        "Others": ["biological", "clothes", "shoes", "trash"]
    }
    
    cols = st.columns(3)
    for i, (category, items) in enumerate(categories.items()):
        with cols[i % 3]:
            st.markdown(f"**{category}:**")
            for item in items:
                st.write(f"• {item}")

def show_prediction_page(app):
    """Display image upload and prediction page"""
    st.markdown("## 📤 Upload Image for Classification")
    
    # Model status check
    if app.model is None:
        st.error("❌ No model loaded. Please load a model first from the sidebar or setup guide.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image of waste material for classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown("**Image Details:**")
            st.write(f"• **Filename:** {uploaded_file.name}")
            st.write(f"• **Size:** {image.size}")
            st.write(f"• **Format:** {image.format}")
        
        with col2:
            st.markdown("### 🎯 Prediction Results")
            
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    pred_class, confidence = app.predict_image(image)
                    
                    if pred_class and confidence:
                        # Main prediction
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>🏷️ Predicted Class</h2>
                            <h1>{pred_class}</h1>
                            <h3>Confidence: {confidence:.2%}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence interpretation
                        if confidence > 0.8:
                            st.success("✅ High confidence prediction!")
                        elif confidence > 0.6:
                            st.warning("⚠️ Moderate confidence prediction")
                        else:
                            st.error("❌ Low confidence prediction")
                        
                        # Top 5 predictions
                        st.markdown("#### 📊 Top 5 Predictions")
                        top_preds = app.get_top_predictions(image, top_k=5)
                        
                        if top_preds:
                            pred_df = pd.DataFrame(top_preds, columns=['Class', 'Confidence'])
                            pred_df['Confidence'] = pred_df['Confidence'].apply(lambda x: f"{x:.2%}")
                            
                            st.dataframe(pred_df, use_container_width=True)
                            
                            # Visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            classes = [pred[0] for pred in top_preds]
                            confidences = [pred[1] for pred in top_preds]
                            
                            bars = ax.barh(classes, confidences, color='skyblue')
                            ax.set_xlabel('Confidence')
                            ax.set_title('Top 5 Predictions')
                            ax.set_xlim(0, 1)
                            
                            # Add confidence labels
                            for bar, conf in zip(bars, confidences):
                                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                                       f'{conf:.2%}', va='center')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    else:
                        st.error("❌ Failed to make prediction. Please try another image.")
    
    else:
        st.info("👆 Please upload an image to get started")
        
        # Usage tips
        st.markdown("""
        <div class="warning-box">
            <strong>💡 Tips for better predictions:</strong>
            <ul>
                <li>Use clear, well-lit images</li>
                <li>Focus on a single waste item</li>
                <li>Avoid cluttered backgrounds</li>
                <li>Supported formats: PNG, JPG, JPEG</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_performance_page(app):
    """Display model performance metrics"""
    st.markdown("## 📊 Model Performance")
    
    # Sample performance metrics
    metrics = {
        "Accuracy": 0.9818,
        "Precision": 0.9829,
        "Recall": 0.9818,
        "F1-Score": 0.9818,
        "AUC-ROC": 0.9999
    }
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎯 Accuracy", f"{metrics['Accuracy']:.2%}")
    with col2:
        st.metric("📏 Precision", f"{metrics['Precision']:.2%}")
    with col3:
        st.metric("📊 Recall", f"{metrics['Recall']:.2%}")
    with col4:
        st.metric("⚖️ F1-Score", f"{metrics['F1-Score']:.2%}")
    with col5:
        st.metric("📈 AUC-ROC", f"{metrics['AUC-ROC']:.4f}")
    
    # Model details
    st.markdown("### 🔧 Model Specifications")
    
    specs_col1, specs_col2 = st.columns(2)
    
    with specs_col1:
        st.markdown("""
        **Architecture:** YOLOv8n-cls
        
        **Input Size:** 224x224 pixels
        
        **Number of Classes:** 22
        
        **Model Size:** ~2.88 MB
        """)
    
    with specs_col2:
        st.markdown("""
        **Training Epochs:** 50
        
        **Batch Size:** 32
        
        **Optimizer:** AdamW
        
        **GitHub Ready:** ✅ Yes
        """)

def show_about_page():
    """Display about page"""
    st.markdown("## ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This Waste Classification System uses advanced computer vision to automatically identify and categorize different types of waste materials. The system is designed to support recycling efforts and improve waste management efficiency.
    
    ### 🧠 Technology Stack
    - **Model:** YOLOv8 Classification (Ultralytics)
    - **Framework:** PyTorch
    - **Frontend:** Streamlit
    - **Image Processing:** OpenCV, PIL
    - **Visualization:** Matplotlib, Seaborn
    - **Deployment:** GitHub + Streamlit Cloud
    
    ### 📋 Supported Categories
    The model can classify 22 different types of waste materials across 6 main categories.
    
    ### 🎯 Model Performance
    - **Accuracy:** 98.18%
    - **Precision:** 98.29%
    - **Recall:** 98.18%
    - **F1-Score:** 98.18%
    - **Model Size:** 2.88 MB (Cloud-ready)
    
    ### 🚀 Deployment Features
    - **GitHub Integration:** Direct model loading from GitHub repositories
    - **Cloud Ready:** Optimized for Streamlit Cloud deployment
    - **Real-time Processing:** Instant image classification
    - **Responsive Design:** Works on desktop and mobile devices
    """)

def show_setup_guide():
    """Display setup guide for GitHub deployment"""
    st.markdown("## 🚀 GitHub Deployment Setup Guide")
    
    st.markdown("### 📁 Required Files in Your GitHub Repository")
    
    files_info = {
        "File": ["waste_classification_app.py", "requirements.txt", "best.pt", "README.md"],
        "Description": [
            "Main Streamlit application code",
            "Python dependencies list",
            "Trained YOLOv8 model weights",
            "Documentation (optional)"
        ],
        "Required": ["✅ Yes", "✅ Yes", "✅ Yes", "⚪ Optional"]
    }
    
    files_df = pd.DataFrame(files_info)
    st.table(files_df)
    
    st.markdown("### 🔗 Getting Your Model URL")
    
    st.markdown("""
    1. **Upload your `best.pt` file to GitHub**
    2. **Navigate to the file in your repository**
    3. **Click on the file name**
    4. **Click the "Raw" button**
    5. **Copy the URL from the address bar**
    
    **Example URL format:**
    ```
    https://github.com/username/repository/raw/main/best.pt
    ```
    """)
    
    st.markdown("### 🌐 Streamlit Cloud Deployment")
    
    st.markdown("""
    1. **Push all files to your GitHub repository**
    2. **Visit [share.streamlit.io](https://share.streamlit.io)**
    3. **Connect your GitHub account**
    4. **Select your repository**
    5. **Set main file path:** `waste_classification_app.py`
    6. **Deploy!**
    
    **Environment Variables (if needed):**
    - No special environment variables required
    - Model will be downloaded automatically from GitHub
    """)
    
    st.markdown("### 🛠️ Local Testing")
    
    st.markdown("""
    ```
    # Clone your repository
    git clone https://github.com/username/your-repo.git
    cd your-repo
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Run the app
    streamlit run waste_classification_app.py
    ```
    """)
    
    st.markdown("### 💡 Pro Tips")
    
    st.markdown("""
    - **Model Size:** Keep your model file under 100MB for GitHub
    - **Performance:** Use GitHub Releases for larger model files
    - **Security:** Never commit API keys or sensitive data
    - **Documentation:** Add a comprehensive README.md
    - **Version Control:** Use Git LFS for large model files if needed
    """)

if __name__ == "__main__":
    main()
