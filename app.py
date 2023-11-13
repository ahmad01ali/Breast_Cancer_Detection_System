import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import time
from reportlab.pdfgen import canvas
from io import BytesIO
from PIL import ImageEnhance, ImageChops

# Load the trained model
model = load_model('breast_cancer_model.h5')

# Create a dictionary to store user details
user_details = {
    'Name': '',
    'Age': 0,
    'Gender': '',
}

# Set the page configuration
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="⚕️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Color palette
primary_color = "#E63946"
secondary_color = "#F1FAEE"
text_color = "#1D3557"
background_color = "#A8DADC"

# Logo image
logo_image = Image.open("new.jpg")

# Styling the sidebar and background color
st.markdown(
    f"""
    <style>
        body {{
            background-color: {background_color};
        }}
        .sidebar .sidebar-content {{
            background-color: {primary_color};
            color: {secondary_color};
        }}
        .css-17eq0hr {{
            background-color: {background_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Display the logo image
st.image(logo_image, use_column_width=False, width=150)

# Page title
st.title("Breast Cancer Detection")

# Collect user details
st.sidebar.subheader("Information")

# Patient Name (Compulsory)
user_details['Name'] = st.sidebar.text_input("Patient Detail:", placeholder="Enter Name")
user_details['sername'] = st.sidebar.text_input("", placeholder="Enter Sername")

if not user_details['Name'] and not user_details['sername']:
    st.warning("Please enter the patient's name.")
    st.stop()

# Patient Age (Compulsory)
user_details['Age'] = st.sidebar.number_input("Patient Age:", max_value=120, placeholder="Enter Age")
if user_details['Age'] == 0:
    st.warning("Please enter a valid age.")
    st.stop()

# Patient Gender (Compulsory)
user_details['Gender'] = st.sidebar.radio("Patient Gender:", ['Female', 'Male'])
if not user_details['Gender']:
    st.warning("Please select the patient's gender.")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the selected image
    # st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image for prediction
    image = Image.open(uploaded_file)
    img_array = np.array(image.resize((64, 64)))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0

    # Check if the image is considered colorful
    def is_colorful_image(img):
        color_enhancer = ImageEnhance.Color(img)
        color_enhanced = color_enhancer.enhance(2.0)  # Enhance color
        difference = ImageChops.difference(img, color_enhanced)
        return difference.getbbox() is not None

    # Function to check if the image is valid for breast cancer detection
    def is_valid_breast_image(img):
        return not is_colorful_image(img)

    # Check aspect ratio to determine if it resembles the shape of a breast
    img = Image.fromarray(img_array[0].astype('uint8'))  # Convert array back to Image object
    width, height = img.size
    width, height = image.size
    aspect_ratio = width / height

    if is_valid_breast_image(image):
        start_time = time.time()  # Start measuring the time
        pred = model.predict(img_array)
        end_time = time.time()  # End measuring the time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        threshold = 0.5
        class_label = "No Cancer" if pred[0][0] >= threshold else "Cancer"

        # Add green box for cancer detection
        if class_label == "Cancer":
            draw = ImageDraw.Draw(image)
            draw.rectangle([(50, 50), (150, 150)], outline="green", width=3)

        # Display the modified image
        st.image(image, caption='Detected Image.', use_column_width=False,width=450)

        # Download report option
        if st.button("Check", key="check_button", use_container_width=True):
            # Generate PDF report
            pdf_filename = f"{user_details['Name']}_breast_cancer_report.pdf"
            pdf_canvas = canvas.Canvas(BytesIO())

            # Add user details to the PDF
            pdf_canvas.drawString(130, 750, f"PATIENT REPORT")
            pdf_canvas.drawString(130, 730, f"Patient Name:{user_details['Name']} {user_details['sername']}")
            pdf_canvas.drawString(130, 710, f"Age:{user_details['Age']}")
            pdf_canvas.drawString(130, 690, f"Gender:{user_details['Gender']}")

            # Add prediction details to the PDF
            pdf_canvas.drawString(130, 590, f"Prediction:{class_label}")
            pdf_canvas.drawString(130, 570, f"Prediction Probability:{pred[0][0]:.2f}")
            pdf_canvas.drawString(130, 550, f"Prediction Time:{elapsed_time:.2f} seconds")

            pdf_canvas.save()

            # Offer the PDF for download
            st.download_button(
                label="Download Report",
                data=BytesIO(pdf_canvas.getpdfdata()),
                file_name=f"{user_details['Name']}_breast_cancer_report.pdf",
                key="report_button", use_container_width=True
            )

            # Read the result using text-to-speech
            engine = pyttsx3.init()
            engine.say(f"Breast cancer detected for {user_details['Name']} {user_details['sername']}." if class_label == "Cancer"
                       else f"No breast cancer detected for {user_details['Name']} {user_details['sername']}.")
            st.subheader(f"Result: {class_label}")
            st.text(f"Required Prediction Time: {elapsed_time:.2f} seconds")
            engine.runAndWait()

    else:
        st.error("Invalid image selected. Please choose a valid breast image.")
