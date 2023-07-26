import streamlit as st
from passlib.hash import pbkdf2_sha256

# Fake database (SQLite can be used in a real application)
# For this example, we'll use a dictionary to store user information
# In a production app, use a real database like SQLite, PostgreSQL, etc.
users = {
    "john": {
        "username": "Daniel",
        "password_hash": pbkdf2_sha256.hash("daniels_password"),
    },
    "jane": {
        "username": "andy",
        "password_hash": pbkdf2_sha256.hash("andys_password"),
    },
}

# Helper function to check if the provided password matches the stored hash
def verify_password(password, password_hash):
    return pbkdf2_sha256.verify(password, password_hash)

# Add login decorator to secure the "Inference" page
def secure_inference_page():
    st.title('Model Inference Section')
    st.write('In this section, you can perform predictions using your trained ML model.')

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
st.write('Thank you for using the ML app. For more information, contact us at contact@example.com.')
