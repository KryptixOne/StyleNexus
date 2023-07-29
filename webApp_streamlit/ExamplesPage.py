# app.py
import streamlit as st

def examplesPage():
    st.title("Discover The Magic")
    st.write('---')
    st.write(
        """
        Discover the captivating world of AI through an array of awe-inspiring examples! Immerse yourself in mesmerizing
         AI-generated artworks, experience the brilliance of natural language processing, and witness the wonders of 
         computer vision applications. Prepare to be amazed by the groundbreaking possibilities that cutting-edge AI 
         technology brings to life in a collection of stunning visuals. Step into the future of innovation and 
         creativity with our extraordinary AI examples!
        """
    )
    st.write('---')
    st.header(
        'Some user examples:'
    )
    col1 , col2, = st.columns(2)


    with col1:
        st.subheader('Original Images')

        st.image('./webApp_streamlit/ExamplePics/ref3.jpg', use_column_width=True, caption='A (sexy) man modeling a sweater with a messy'
                                                                                 ' background')

        st.image('./webApp_streamlit/ExamplePics/GirlWearingLion.PNG', use_column_width=True,
                 caption='A woman wearing a unique lion graphic Tee. Standard-non-unique mockup from "printify.com"')

    with col2:
        st.subheader('Generated Fashion Models')
        st.image('./webApp_streamlit/ExamplePics/ref1_to_model.png', use_column_width=True, caption='AI Generated Fashion Model '
                                                                                          'wearing a silly unicorn sweater')

        st.image('./webApp_streamlit/ExamplePics/girlToNewGirl.PNG', use_column_width=True,
                 caption='A fashion Model wearing a unique lion graphic Tee. Custom Generated AI Fashion Model ')


    st.write('Like what you see? Give it a try! [Demo](#demo)')