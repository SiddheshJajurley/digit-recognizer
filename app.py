import streamlit as st
import numpy as np
from PIL import Image
import pickle

# Set title
st.title("🖋️ Handwritten Digit Recognizer (SVM)")

st.write("Upload a handwritten digit image (8x8 grayscale image from sklearn's digits dataset).")

# Load the pickled SVM model and scaler
@st.cache_resource
def load_model():
    with open('svm_digit_model.pkl', 'rb') as f:
        model, scaler = pickle.load(f)
    return model, scaler

clf, scaler = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to numpy array
    img_array = np.array(image)

    # Display raw array (optional)
    st.write("Pixel values (after loading):")
    st.write(img_array)

    # Check if the image is 8x8s
    if img_array.shape != (8, 8):
        st.error("❌ The uploaded image is not 8x8. Please upload an 8x8 grayscale image (like from load_digits).")
    else:
        # Flatten & scale (from 0-255 to 0-16 to match sklearn digits dataset)
        img_scaled = (img_array / 255.0) * 16.0
        img_flat = img_scaled.flatten().reshape(1, -1)

        # Predict button
        if st.button("Predict"):
            # First scale using StandardScaler
            img_transformed = scaler.transform(img_flat)
            prediction = clf.predict(img_transformed)[0]
            st.success(f"✅ Predicted Digit: **{prediction}**")
