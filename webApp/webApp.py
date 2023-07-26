import streamlit as st
from PIL import Image
from io import BytesIO
from passlib.hash import sha256_crypt

# Fake database (SQLite can be used in a real application)
# For this example, we'll use a dictionary to store user information
# In a production app, use a real database like SQLite, PostgreSQL, etc.
users = {
    "john": {
        "username": "Daniel",
        "password_hash": sha256_crypt.hash("daniels_password"),
    },
    "jane": {
        "username": "andy",
        "password_hash": sha256_crypt.hash("andys_password"),
    },
}

# Helper function to check if the provided password matches the stored hash
def verify_password(password, password_hash):
    return sha256_crypt.verify(password, password_hash)

# Function to perform ML model inference on the uploaded image
def perform_inference(image):
    # Your ML model inference code here
    # Replace this with your actual ML model code
    # For demonstration purposes, let's return a fixed example prediction
    return "Prediction: Dress"  # Replace this with the actual model's output image

# Function to downsample the image to a maximum resolution of 256x256
def downsample_image(image):
    max_size = (256, 256)
    image.thumbnail(max_size, Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
    return image

# Add login decorator to secure the "Inference" page
def secure_inference_page():
    st.title('Model Inference Section')
    st.write('In this section, you can perform predictions using your trained ML model.')

    # File uploader to allow users to upload images
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to PIL image
        pil_image = Image.open(uploaded_file)

        # Downsample the image
        downsampled_image = downsample_image(pil_image)

        # Display the input and output images side by side in two columns
        col1, col2 = st.columns(2)
        prediction = None

        if st.button("Generate"):
            prediction = perform_inference(pil_image)

        with col1:
            # Display the downsampled input image on the webpage
            st.image(downsampled_image, caption="Downsampled Input Image", use_column_width=True)

        with col2:
            # Perform inference on the original resolution image
                # Display the output of the ML model
                st.image(pil_image, caption=prediction, use_column_width=True)

    # Add empty space for the "Generate" button below the displayed images
    st.empty()

# Set page config
st.set_page_config(page_title="StyleNexus Homepage", page_icon=':rocket:', layout='wide')

# Sidebar with app navigation options
st.sidebar.title('Navigation')
selected_page = st.sidebar.radio('Go to', ['Home', 'Create your Fashion Models', 'Examples'])

# Main title and description
st.title('Welcome to ML App')
st.write('This is a simple homepage for your ML app.')

# Set this variable to control whether login is required or not
login_required = False  # Set to True to require login, or False to bypass login

# Check if the user is logged in before displaying the Inference page content
if selected_page == 'Create your Fashion Models':
    if login_required:
        st.subheader('Login')
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in users and verify_password(password, users[username]["password_hash"]):
                secure_inference_page()
            else:
                st.warning("Invalid username or password. Please try again.")
    else:
        secure_inference_page()

# Examples page content
elif selected_page == 'Examples':
    st.title('Examples')
    st.write('This page showcases some example use cases or demonstrations of your ML model predictions.')

    # Add your example content here
    st.subheader('Example 1')
    st.write('Description of example 1.')
    #st.image('example1.png', use_column_width=True)

    st.subheader('Example 2')
    st.write('Description of example 2.')
    #st.image('example2.png', use_column_width=True)

# Footer
st.markdown('---')
st.write('Thank you for using the ML app.')
