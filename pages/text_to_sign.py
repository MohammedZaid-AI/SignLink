import streamlit as st
import os
import time

# --- Configuration ---
# This is the folder where your A.png, B.png, etc. images are.
IMAGE_DIR = "sign_images" 
# How long to show each letter, in seconds.
TIME_PER_LETTER = 0.7 

st.title("Text-to-Fingerspelling App 🗣️➡️🖐️")

# --- Text-to-Sign (Fingerspelling) Function ---

def display_fingerspelling(text, image_placeholder):
    """
    Displays the sign language images for each letter in the text.
    """
    # Convert text to uppercase
    text = text.upper()
    
    st.info(f"Fingerspelling: {text}")
    
    for char in text:
        if 'A' <= char <= 'Z':
            image_path = os.path.join(IMAGE_DIR, f"{char}.jpg")
        elif char == ' ':
            image_path = os.path.join(IMAGE_DIR, "space.jpg")
        else:
            # Skip characters we don't have images for (like '?' or '!')
            continue 

        # Check if the image file actually exists
        if os.path.exists(image_path):
            image_placeholder.image(image_path, width=300)
            # Pause to create the animation effect
            time.sleep(TIME_PER_LETTER)
        else:
            st.warning(f"Warning: Image file not found for '{char}'")
            # Show a blank space even if the file is missing
            image_placeholder.empty() 
            time.sleep(TIME_PER_LETTER)
        
    # Clear the image at the end of the word
    time.sleep(1)
    image_placeholder.empty()
    st.success("Fingerspelling complete!")

# --- Main App Layout ---

st.header("Translate English to Sign")
st.write("Type a word or sentence, and the app will fingerspell it using sign language images.")

# 1. This is the "screen" where the avatar's hands will appear.
avatar_placeholder = st.empty()

# 2. Text Input
user_text = st.text_input("Type your message here:")

# 3. "Show Signs" Button
if st.button("Show Signs"):
    if user_text:
        # Call the function to run the animation
        display_fingerspelling(user_text, avatar_placeholder)
    else:
        st.warning("Please type a message first.")