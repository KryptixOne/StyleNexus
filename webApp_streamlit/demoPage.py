# app.py
import streamlit as st

def demo():

    st.title("Give It A Try!")
    st.write("All you need is:\n" 
             "1) A picture of yourself wearing or holding whatever item you wish your model to be showing off.\n"
             "2) You describe the item you are trying to sell (less detail is more) \n"
             "3) Specify what the fashion model you want.\n"
             "4) Then you press the generate button and let the magic unfold!\n\n"
             
             "It really is just that simple!")


    # Add interactive elements, e.g., a button and a text input
    name = st.text_input("Enter your name", "John Doe")
    st.write(f"Hello, {name}!")

    if st.button("Click me!"):
        st.write("You clicked the button!")

