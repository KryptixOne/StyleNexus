import streamlit as st
from homePage import homepage
from demoPage import demo
from ExamplesPage import examplesPage
from aboutPage import aboutPage


def main():
    st.set_page_config(
        page_title="Style Nexus",
        page_icon="🧊",
        layout="centered",
        initial_sidebar_state="expanded",

    )
    #Set Sidebar width
    css_sidebar_width = '''
            <style>
                [data-testid="stSidebar"] {
                    min-width: 300px;
                    max-width: 400px;
                }
            </style>
        '''

    st.markdown(css_sidebar_width, unsafe_allow_html=True)

    # Set Page Width
    css = '''
    <style>
        section.main > div {max-width:75rem}
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)



    st.sidebar.title('Navigation')
    page = st.sidebar.radio("Choose a page", ["Homepage", 'About Us',"Examples", "Demo"])

    if page == 'Homepage':
        homepage()
    elif page =='About Us':
        aboutPage()
    elif page == 'Examples':
        examplesPage()
    elif page == 'Demo':
        demo()


if __name__ == "__main__":
    main()