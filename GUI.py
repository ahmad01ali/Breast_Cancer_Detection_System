







import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
from PIL import ImageEnhance, ImageChops

# Load the trained model
model = load_model('breast_cancer_model.h5')

# Initialize img_array as a global variable
img_array = None


# Function to open the image file
def open_image():
    global img_array  # Access the global img_array variable
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((64, 64))  # Resize to the input shape required by the model

        if is_colorful_image(img):
            show_error_message("No valid image selected.")
        else:
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32') / 255.0

            img = ImageTk.PhotoImage(img)
            image_label.configure(image=img)
            image_label.image = img
            result_label.config(text="")  # Reset the prediction result
    else:
        show_error_message("No valid image selected.")


# Function to predict the image
def predict_image():
    global img_array  # Access the global img_array variable

    if img_array is not None and img_array.shape == (1, 64, 64, 3):
        img = Image.fromarray(img_array[0].astype('uint8'))  # Convert array back to Image object
        width, height = img.size
        aspect_ratio = width / height

        # Check aspect ratio to determine if it resembles the shape of a breast
        if 0.8 <= aspect_ratio <= 1.2:
            start_time = time.time()  # Start measuring the time
            pred = model.predict(img_array)
            end_time = time.time()  # End measuring the time
            elapsed_time = end_time - start_time  # Calculate the elapsed time
            threshold = 0.5
            class_label = "No Cancer" if pred[0][0] >= threshold else "Cancer"
            result_label.config(text=f"{class_label} Detected", fg="red" if class_label == "Cancer" else "green")

            # Read the result using text-to-speech
            read_result(class_label)

            # Display the elapsed time
            result_label_time.config(text=f"Required Prediction Time: {elapsed_time:.2f} seconds" ,fg="blue")
        else:
            show_error_message("No valid image selected.")
    else:
        show_error_message("No valid image selected.")


# Function to read the result using text-to-speech
def read_result(class_label):
    engine = pyttsx3.init()

    # Set the voice to a female voice
    voices = engine.getProperty('voices')
    female_voice = next((voice for voice in voices if "female" in voice.name.lower()), None)
    if female_voice:
        engine.setProperty('voice', female_voice.id)

    # Adjust the speaking rate
    engine.setProperty('rate', 150)  # Adjust the value as needed

    if img_array is not None and img_array.shape == (1, 64, 64, 3):
        img = Image.fromarray(img_array[0].astype('uint8'))  # Convert array back to Image object
        width, height = img.size
        aspect_ratio = width / height
        engine.say("Breast cancer detected." if class_label == "Cancer" else "No breast cancer detected.")
    elif img_array is not None and img_array.shape != (1, 64, 64, 3):
        engine.say("Invalid image selected for unknown result")
    engine.runAndWait()


# Function to remove the image
def remove_image():
    global img_array
    img_array = None
    image_label.config(image="")
    result_label.config(text="")
    result_label_time.config(text="")  # Reset the elapsed time


# Function to check if the image is considered colorful
def is_colorful_image(img):
    color_enhancer = ImageEnhance.Color(img)
    color_enhanced = color_enhancer.enhance(2.0)  # Enhance color
    difference = ImageChops.difference(img, color_enhanced)
    return difference.getbbox() is not None


# Function to show error messages
def show_error_message(message):
    result_label.config(text=message, fg="black")
    result_label_time.config(text="")  # Reset the elapsed time


# Create the main window
window = tk.Tk()
window.title("Breast Cancer Detection")
window.geometry("400x600")  # Increased the height to accommodate larger images
window.configure(bg="#d6cadd")

# Create title label with blue color (hexadecimal code: #236B8E)
title_label = tk.Label(window, text="Welcome To Breast Cancer Detection", font=("Arial", 16), fg="#990000",
                       bg="#d6cadd")
title_label.pack(pady=10)

# Create image label
image_label = tk.Label(window, bg="#d6cadd")
image_label.pack(pady=10)

# Create button toopen the image
open_button = tk.Button(window, text="Open Image", command=open_image, font=("Arial", 14))
open_button.pack(pady=10)

# Create button to remove the image
remove_button = tk.Button(window, text="Remove Image", command=remove_image, font=("Arial", 14))
remove_button.pack(pady=10)

# Create button to predict the image with a unique color (hexadecimal code: #FF8C00)
predict_button = tk.Button(window, text="CHECK", command=predict_image, bg="#a1caf1", font=("Arial", 14))
predict_button.pack(pady=10)

# Create label to display the prediction result
result_label = tk.Label(window, text="", font=("Arial", 14), bg="#d6cadd")
result_label.pack(pady=10)

# Create label to display the elapsed time
result_label_time = tk.Label(window, text="", font=("Arial", 12), bg="#d6cadd")
result_label_time.pack(pady=5)

# Run the GUI main loop
window.mainloop()
