import streamlit as st

st.set_page_config(
    page_title="SignLink: Main Menu",
    page_icon="👋",
)

st.title("Welcome to SignLink! 🌉")
st.write("AI-Powered Communication for All")

st.markdown(
    """
    This is your two-way communication bridge. Please choose what
    you would like to do from the menu on the left.

    ### Features:
    
    * **Sign-to-Speech:** Use this if you are a signer. The app will watch your 
        hand signs through the webcam and speak the words you form.
    
    * **Text-to-Sign:** Use this if you are a speaker/typer. Type a message, 
        and the app will show the fingerspelling for each letter.
    """
)

st.sidebar.success("Select a tool above to start.")