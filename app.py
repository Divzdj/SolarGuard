import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
# Import necessary layers for model reconstruction
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

# -----------------------------------------------------------------------------
# --- 1. Configuration Constants ---
# -----------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
CLASSES = ["Clean", "Dusty", "Bird-drop", "Electrical-damage", "Physical-Damage", "Snow-Covered"]
# CRITICAL CORRECTION: MUST use the .weights.h5 extension as required by the Keras
# ModelCheckpoint API when using save_weights_only=True.
MODEL_PATH = 'best_classification_weights.weights.h5' 
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = len(CLASSES)

# -----------------------------------------------------------------------------
# --- 2. Model Loading (Cached for Efficiency) ---
# -----------------------------------------------------------------------------
@st.cache_resource
def load_best_model():
    """
    Rebuilds the model structure and loads weights.
    This bypasses the saving/loading error encountered with .h5 and .keras 
    for complex MobileNetV2 structures.
    """
    try:
        # 2a. Rebuild the exact model structure from SolarGuard.ipynb
        # Load base model with pre-trained ImageNet weights, excluding the top classifier
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
        base_model.trainable = False # Must match the frozen state used during training

        # Create the sequential model exactly as defined in the notebook
        model = Sequential([
            base_model,
            Flatten(), # The layer that caused the previous errors
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation="softmax")
        ])
        
        # 2b. Compile the model (necessary before loading weights)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # 2c. Load the saved weights (not the full model structure)
        model.load_weights(MODEL_PATH)
        
        # 2d. Force a prediction to finalize graph building (prevents first-run errors)
        dummy_input = np.zeros((1, *IMAGE_SIZE, 3), dtype=np.float32)
        model.predict(dummy_input, verbose=0)

        return model
        
    except Exception as e:
        st.error(f"Error: Could not load the model weights from {MODEL_PATH}.")
        st.error("Ensure the file 'best_classification_weights.weights.h5' is in the same directory as app.py.")
        st.error("Also, ensure you ran the Colab notebook to generate this file.")
        st.error(f"Detailed Error: {e}")
        return None

# -----------------------------------------------------------------------------
# --- 3. Prediction Logic ---
# -----------------------------------------------------------------------------
def predict_image(image, model):
    """Preprocess the PIL image, make a prediction, and return the result."""
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize image data [0, 1]
    img_array = img_array / 255.0

    predictions = model.predict(img_array, verbose=0)
    
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASSES[predicted_index]
    confidence = np.max(predictions)

    return predicted_class, confidence

# -----------------------------------------------------------------------------
# --- 4. Streamlit UI and Execution ---
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="SolarGuard Defect Classifier ", layout="wide")
    st.title("SolarGuard: AI-Powered Solar Panel Defect Classifier")
    st.markdown("---")
    
    model = load_best_model()
    if model is None:
        return # Stop if model loading failed

    # Image Upload Widget
    uploaded_file = st.file_uploader(" Choose an image file...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
            st.subheader("Analysis Result")
        
        # Prediction Logic
        if st.button('✨ Analyze Panel Condition'):
            with st.spinner('Analyzing panel and generating prediction...'):
                predicted_class, confidence = predict_image(image, model)

                with col2:
                    st.success(f"Analysis Complete! ")
                    st.metric(label="Predicted Condition", value=predicted_class)
                    st.info(f"Confidence: {confidence*100:.2f}%")

                    st.markdown("### Actionable Business Insight")
                    
                    if predicted_class == 'Clean':
                        st.balloons()
                        st.write(" **Recommendation:** Panel is operating optimally. No immediate action required.")
                    elif predicted_class in ['Dusty', 'Bird-drop', 'Snow-Covered']:
                        st.warning(" **Recommendation:** Surface contamination detected. Schedule a **Low-Priority Cleaning** dispatch to restore maximum efficiency.")
                    elif predicted_class == 'Physical-Damage':
                        st.error(" **Recommendation (HIGH PRIORITY):** Potential Physical-Damage detected. Dispatch a **Tier 1 Drone Inspection** first to confirm before costly physical repair is scheduled.")
                    elif predicted_class == 'Electrical-damage':
                        st.error(" **Recommendation (CRITICAL):** Electrical damage is suspected. Schedule an **Urgent Technician Safety Inspection** immediately.")
                    else:
                        st.write("No specific insight available for this classification.")

if __name__ == '__main__':
    main()