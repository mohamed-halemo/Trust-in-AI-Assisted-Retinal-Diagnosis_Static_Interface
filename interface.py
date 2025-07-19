import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
from datetime import datetime
import random
import os

# Page configuration
st.set_page_config(
    page_title="DR Stage Prediction System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .trust-score-high {
        color: #28a745;
        font-weight: bold;
    }
    .trust-score-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .trust-score-low {
        color: #dc3545;
        font-weight: bold;
    }
    .feedback-form {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .trust-calculation {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {}
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []

# Static DR stage probabilities (0-4 scale)
STATIC_DR_PROBABILITIES = {
    'No DR (0)': 0.15,
    'Mild NPDR (1)': 0.25,
    'Moderate NPDR (2)': 0.35,
    'Severe NPDR (3)': 0.20,
    'PDR (4)': 0.05
}

# Trust Score Calculation Parameters
TRUST_WEIGHTS = {
    'w1': 0.35,  # Model confidence weight
    'w2': 0.25,  # Image quality score weight
    'w3': 0.20,  # Clinician agreement weight
    'w4': 0.20   # Historical consistency weight
}

def calculate_trust_score(confidence, image_quality, clinician_agreement, historical_consistency):
    """
    Calculate trust score using the formula:
    Trust Score = w1*C + w2*Q + w3*A + w4*H
    
    Where:
    C = Model confidence
    Q = Image quality score
    A = Clinician agreement or override
    H = Historical similarity-based consistency score
    """
    trust_score = (
        TRUST_WEIGHTS['w1'] * confidence +
        TRUST_WEIGHTS['w2'] * image_quality +
        TRUST_WEIGHTS['w3'] * clinician_agreement +
        TRUST_WEIGHTS['w4'] * historical_consistency
    )
    return min(1.0, max(0.0, trust_score))  # Ensure between 0 and 1

def generate_static_prediction():
    """Generate static DR stage prediction with fixed probabilities"""
    # Static values for demonstration - these will be the same for consistency
    confidence = 0.87
    image_quality = 0.92
    clinician_agreement = 0.85  # 1.0 if no override, 0.5 if override
    historical_consistency = 0.78
    
    # Calculate trust score
    trust_score = calculate_trust_score(confidence, image_quality, clinician_agreement, historical_consistency)
    
    # Determine the predicted stage based on highest probability
    predicted_stage = max(STATIC_DR_PROBABILITIES, key=STATIC_DR_PROBABILITIES.get)
    
    return {
        'stage': predicted_stage,
        'confidence': confidence,
        'trust_score': trust_score,
        'stage_probs': STATIC_DR_PROBABILITIES,
        'image_quality': image_quality,
        'clinician_agreement': clinician_agreement,
        'historical_consistency': historical_consistency
    }

def load_static_gradcam():
    """Load the static Grad-CAM image from images folder"""
    try:
        gradcam_path = os.path.join('images', 'gradcam.png')
        if os.path.exists(gradcam_path):
            return Image.open(gradcam_path)
        else:
            st.error(f"Grad-CAM image not found at {gradcam_path}")
            return None
    except Exception as e:
        st.error(f"Error loading Grad-CAM image: {e}")
        return None

def create_simplified_heatmap(image, prediction):
    """Create simplified heatmap for patient view"""
    # Create a simpler, more patient-friendly visualization
    img_array = np.array(image.convert('RGB'))
    height, width = img_array.shape[:2]
    
    # Create a simple overlay with fewer, larger areas
    heatmap = np.zeros((height, width))
    
    # Add 1-2 main attention areas
    for _ in range(random.randint(1, 2)):
        center_x = random.randint(width//4, 3*width//4)
        center_y = random.randint(height//4, 3*height//4)
        radius = random.randint(40, 100)
        
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        heatmap[mask] = random.uniform(0.4, 0.7)
    
    # Normalize and smooth
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    # Create overlay with softer colors
    overlay = np.zeros_like(img_array)
    overlay[:, :, 1] = heatmap * 200  # Green channel for softer look
    overlay[:, :, 2] = heatmap * 100  # Blue channel
    
    # Blend with original image
    alpha = 0.4
    blended = img_array * (1 - alpha) + overlay * alpha
    blended = blended.astype(np.uint8)
    
    return Image.fromarray(blended)

def get_trust_score_color(score):
    """Get color class for trust score"""
    if score >= 0.8:
        return "trust-score-high"
    elif score >= 0.6:
        return "trust-score-medium"
    else:
        return "trust-score-low"

def display_trust_calculation(prediction):
    """Display the trust score calculation breakdown"""
    st.markdown('<h3 class="sub-header">🔍 Trust Score Calculation</h3>', unsafe_allow_html=True)
    
    # Create a container with custom styling
    with st.container():
        st.markdown("""
        <div class="trust-calculation">
            <h4>Trust Score Formula:</h4>
            <p><strong>Trust Score = w₁×C + w₂×Q + w₃×A + w₄×H</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Component Breakdown:**")
        
        # Use Streamlit columns for better layout
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**Component**")
        with col2:
            st.markdown("**Value × Weight**")
        with col3:
            st.markdown("**Result**")
        
        # Component calculations
        c_result = prediction['confidence'] * TRUST_WEIGHTS['w1']
        q_result = prediction['image_quality'] * TRUST_WEIGHTS['w2']
        a_result = prediction['clinician_agreement'] * TRUST_WEIGHTS['w3']
        h_result = prediction['historical_consistency'] * TRUST_WEIGHTS['w4']
        
        # Display each component
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("C (Model Confidence)")
        with col2:
            st.markdown(f"{prediction['confidence']:.1%} × {TRUST_WEIGHTS['w1']:.2f}")
        with col3:
            st.markdown(f"**{c_result:.3f}**")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("Q (Image Quality)")
        with col2:
            st.markdown(f"{prediction['image_quality']:.1%} × {TRUST_WEIGHTS['w2']:.2f}")
        with col3:
            st.markdown(f"**{q_result:.3f}**")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("A (Clinician Agreement)")
        with col2:
            st.markdown(f"{prediction['clinician_agreement']:.1%} × {TRUST_WEIGHTS['w3']:.2f}")
        with col3:
            st.markdown(f"**{a_result:.3f}**")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("H (Historical Consistency)")
        with col2:
            st.markdown(f"{prediction['historical_consistency']:.1%} × {TRUST_WEIGHTS['w4']:.2f}")
        with col3:
            st.markdown(f"**{h_result:.3f}**")
        
        st.markdown("---")
        
        # Final calculation
        final_sum = c_result + q_result + a_result + h_result
        trust_color = get_trust_score_color(prediction['trust_score'])
        
        st.markdown("**Final Trust Score:**")
        st.markdown(f"""
        <div class="trust-calculation">
            <p><strong>{c_result:.3f} + {q_result:.3f} + {a_result:.3f} + {h_result:.3f} = <span class="{trust_color}">{prediction['trust_score']:.1%}</span></strong></p>
        </div>
        """, unsafe_allow_html=True)

def clinician_dashboard():
    """Clinician Dashboard Interface"""
    st.markdown('<h1 class="main-header">👨‍⚕️ Clinician Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for patient selection
    st.sidebar.header("Patient Management")
    
    # Mock patient list
    patients = [
        {"id": "P001", "name": "John Smith", "age": 65, "last_visit": "2024-01-15"},
        {"id": "P002", "name": "Mary Johnson", "age": 58, "last_visit": "2024-01-10"},
        {"id": "P003", "name": "Robert Davis", "age": 72, "last_visit": "2024-01-12"},
    ]
    
    selected_patient = st.sidebar.selectbox(
        "Select Patient",
        patients,
        format_func=lambda x: f"{x['name']} ({x['id']})"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Patient ID:** {selected_patient['id']}")
    st.sidebar.markdown(f"**Age:** {selected_patient['age']}")
    st.sidebar.markdown(f"**Last Visit:** {selected_patient['last_visit']}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📸 Retinal Image Analysis</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Retinal Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a retinal fundus image for DR stage prediction"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Retinal Image", use_column_width=True)
            
            # Generate static prediction
            prediction = generate_static_prediction()
            
            # Display prediction results
            st.subheader("AI Prediction Results")
            
            # Info box for static demonstration
            st.info("📊 **Demonstration Mode**: This shows static example values for demonstration purposes. In a real system, these would be calculated by the AI model.")
            
            # Create metrics display
            col1_1, col1_2, col1_3 = st.columns(3)
            
            with col1_1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Predicted Stage</h4>
                    <h3>{prediction['stage']}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col1_2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Confidence</h4>
                    <h3>{prediction['confidence']:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col1_3:
                trust_color = get_trust_score_color(prediction['trust_score'])
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Trust Score</h4>
                    <h3 class="{trust_color}">{prediction['trust_score']:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Display trust score calculation
            display_trust_calculation(prediction)
            
            # Stage probability distribution
            st.subheader("Stage Probability Distribution (0-4 Scale)")
            stage_df = pd.DataFrame(list(prediction['stage_probs'].items()), 
                                  columns=['Stage', 'Probability'])
            
            fig = px.bar(stage_df, x='Stage', y='Probability', 
                        color='Probability', color_continuous_scale='RdYlBu_r')
            fig.update_layout(height=400, xaxis_title="DR Stage (0-4)", yaxis_title="Probability")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">🔍 Grad-CAM Analysis</h2>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Load static Grad-CAM image
            gradcam_img = load_static_gradcam()
            
            if gradcam_img:
                st.image(gradcam_img, caption="Grad-CAM Attention Map (Static Example)", use_column_width=True)
                
                st.markdown("""
                **Grad-CAM Interpretation:**
                - Red areas indicate regions the AI focused on
                - Brighter red = higher attention
                - These areas influenced the prediction most
                - This is a static example for demonstration
                """)
            else:
                st.warning("Grad-CAM image not available")
        
        st.markdown('<h2 class="sub-header">✏️ Correction Tools</h2>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Correction interface
            st.markdown("**Manual Override**")
            
            corrected_stage = st.selectbox(
                "Correct Prediction",
                ['No DR (0)', 'Mild NPDR (1)', 'Moderate NPDR (2)', 'Severe NPDR (3)', 'PDR (4)'],
                index=2  # Default to Moderate NPDR (2)
            )
            
            correction_reason = st.text_area(
                "Reason for Correction",
                placeholder="Explain why you're correcting the AI prediction..."
            )
            
            if st.button("Save Correction"):
                st.success("Correction saved! This will help improve the AI model.")
                
                # Store correction data
                correction_data = {
                    'patient_id': selected_patient['id'],
                    'original_prediction': prediction['stage'],
                    'corrected_stage': corrected_stage,
                    'reason': correction_reason,
                    'timestamp': datetime.now().isoformat()
                }
                
                if 'corrections' not in st.session_state:
                    st.session_state.corrections = []
                st.session_state.corrections.append(correction_data)

def patient_view():
    """Patient View Interface"""
    st.markdown('<h1 class="main-header">👤 Patient Portal</h1>', unsafe_allow_html=True)
    
    # Patient-friendly explanation
    st.markdown("""
    <div style="background-color: #e8f4fd; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem;">
        <h2 style="color: #1f77b4;">Welcome to Your Eye Health Portal</h2>
        <p style="font-size: 1.1rem; line-height: 1.6;">
            This AI system helps your doctor understand your eye health by analyzing images of your retina. 
            It looks for signs of diabetic retinopathy, a condition that can affect people with diabetes.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📸 Your Retinal Image</h2>', unsafe_allow_html=True)
        
        # File uploader for patient
        uploaded_file = st.file_uploader(
            "Upload Your Retinal Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload the retinal image your doctor took"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Display original image
            st.image(image, caption="Your Retinal Image", use_column_width=True)
            
            # Generate static prediction
            prediction = generate_static_prediction()
            
            # Patient-friendly results
            st.markdown('<h2 class="sub-header">📊 Your Results</h2>', unsafe_allow_html=True)
            
            # Info box for static demonstration
            st.info("📊 **Demonstration Mode**: This shows example values for demonstration purposes. In a real system, these would be calculated by the AI model.")
            
            # Simple explanation
            stage_explanations = {
                'No DR (0)': 'No signs of diabetic retinopathy detected. Your eyes appear healthy.',
                'Mild NPDR (1)': 'Early signs of diabetic retinopathy. Regular monitoring recommended.',
                'Moderate NPDR (2)': 'Moderate signs of diabetic retinopathy. Treatment may be needed.',
                'Severe NPDR (3)': 'Significant signs of diabetic retinopathy. Treatment likely required.',
                'PDR (4)': 'Advanced diabetic retinopathy. Immediate treatment recommended.'
            }
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4;">
                <h3>AI Assessment: {prediction['stage']}</h3>
                <p style="font-size: 1.1rem;">{stage_explanations.get(prediction['stage'], 'Please consult your doctor for detailed explanation.')}</p>
                <p><strong>Confidence Level:</strong> {prediction['confidence']:.1%}</p>
                <p><strong>Trust Score:</strong> {prediction['trust_score']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">🔍 Simplified Analysis</h2>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Show simplified heatmap if patient opts in
            show_heatmap = st.checkbox("Show me what the AI is looking at (simplified view)")
            
            if show_heatmap:
                simplified_img = create_simplified_heatmap(image, prediction)
                st.image(simplified_img, caption="Areas of Interest (Simplified)", use_column_width=True)
                
                st.markdown("""
                **What you're seeing:**
                - Green areas show where the AI focused
                - These areas helped determine your result
                - This is a simplified view for your understanding
                """)
        
        # Feedback form
        st.markdown('<h2 class="sub-header">💬 Your Feedback</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feedback-form">
            <h4>Help us improve!</h4>
            <p>Your feedback helps make this system better for everyone.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Feedback questions
            st.markdown("**Did this explanation help you understand your condition?**")
            helpful_rating = st.radio(
                "Rate your understanding:",
                ["Not helpful at all", "Somewhat helpful", "Very helpful", "Extremely helpful"],
                horizontal=True
            )
            
            st.markdown("**Would you like to help improve the system?**")
            allow_usage = st.radio(
                "Can we use your image (anonymously) to improve the AI?",
                ["No, thank you", "Yes, I'd like to help"],
                horizontal=True
            )
            
            additional_feedback = st.text_area(
                "Any additional comments or suggestions?",
                placeholder="Tell us how we can make this better for patients..."
            )
            
            if st.button("Submit Feedback"):
                # Store feedback
                feedback_data = {
                    'timestamp': datetime.now().isoformat(),
                    'helpful_rating': helpful_rating,
                    'allow_usage': allow_usage,
                    'additional_feedback': additional_feedback,
                    'prediction_stage': prediction['stage'] if uploaded_file else None
                }
                
                if 'patient_feedback' not in st.session_state:
                    st.session_state.patient_feedback = []
                st.session_state.patient_feedback.append(feedback_data)
                
                st.success("Thank you for your feedback! It will help us improve the system.")

def admin_panel():
    """Admin Panel for System Monitoring"""
    st.markdown('<h1 class="main-header">⚙️ System Administration</h1>', unsafe_allow_html=True)
    
    # System statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", "1,247")
    with col2:
        st.metric("Average Trust Score", "78.5%")
    with col3:
        st.metric("Corrections Made", "23")
    with col4:
        st.metric("Patient Feedback", "156")
    
    # Trust Score Components Analysis
    st.markdown('<h2 class="sub-header">Trust Score Component Analysis</h2>', unsafe_allow_html=True)
    
    # Create a sample prediction for demonstration
    sample_prediction = generate_static_prediction()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Component Weights")
        weights_df = pd.DataFrame(list(TRUST_WEIGHTS.items()), columns=['Component', 'Weight'])
        fig = px.pie(weights_df, values='Weight', names='Component', 
                    title="Trust Score Component Weights")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Current Component Values")
        components = {
            'Model Confidence': sample_prediction['confidence'],
            'Image Quality': sample_prediction['image_quality'],
            'Clinician Agreement': sample_prediction['clinician_agreement'],
            'Historical Consistency': sample_prediction['historical_consistency']
        }
        components_df = pd.DataFrame(list(components.items()), columns=['Component', 'Value'])
        fig = px.bar(components_df, x='Component', y='Value', 
                    title="Current Component Values")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent corrections
    st.markdown('<h2 class="sub-header">Recent Corrections</h2>', unsafe_allow_html=True)
    
    if 'corrections' in st.session_state and st.session_state.corrections:
        corrections_df = pd.DataFrame(st.session_state.corrections)
        st.dataframe(corrections_df, use_container_width=True)
    else:
        st.info("No corrections recorded yet.")
    
    # Patient feedback summary
    st.markdown('<h2 class="sub-header">Patient Feedback Summary</h2>', unsafe_allow_html=True)
    
    if 'patient_feedback' in st.session_state and st.session_state.patient_feedback:
        feedback_df = pd.DataFrame(st.session_state.patient_feedback)
        
        # Feedback statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Understanding Ratings")
            helpful_counts = feedback_df['helpful_rating'].value_counts()
            fig = px.pie(values=helpful_counts.values, names=helpful_counts.index)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Usage Permission")
            usage_counts = feedback_df['allow_usage'].value_counts()
            fig = px.bar(x=usage_counts.index, y=usage_counts.values)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No patient feedback recorded yet.")

def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    page = st.sidebar.selectbox(
        "Choose Interface",
        ["Clinician Dashboard", "Patient View", "Admin Panel"]
    )
    
    # Display selected page
    if page == "Clinician Dashboard":
        clinician_dashboard()
    elif page == "Patient View":
        patient_view()
    elif page == "Admin Panel":
        admin_panel()

if __name__ == "__main__":
    main()
