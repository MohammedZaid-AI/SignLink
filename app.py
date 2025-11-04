import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import mediapipe as mp
import math

# --- 1. Load Model and Labels ---
MODEL_PATH = 'best_model.h5'
LABELS_PATH = 'labels.json'

# Load the TFLite model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load the labels
try:
    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)
    print(f"Loaded {len(labels)} labels.")
except Exception as e:
    st.error(f"Error loading labels: {e}")
    st.stop()

# --- 2. Mediapipe Setup ---
# This is to find the hand's bounding box
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# --- 3. Helper Function ---
def get_hand_bounding_box(hand_landmarks, image_shape, padding=20):
    """Calculates the bounding box for the hand with padding."""
    h, w, _ = image_shape
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(h, int(max(y_coords)) + padding)
    
    return [x_min, y_min, x_max, y_max]

# --- 4. Streamlit App ---
st.title("Sign Language CNN Detector 📸")
st.write("Hold a sign, and the model will try to guess it!")

frame_placeholder = st.empty()
result_placeholder = st.empty()
stop_button = st.button("Stop Webcam")

# --- 5. Main Webcam Loop ---
cap = cv2.VideoCapture(0)

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.write("Webcam feed ended.")
        break
        
    frame = cv2.flip(frame, 1)  # Flip horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with Mediapipe
    results = hands.process(frame_rgb)
    
    # Create a copy to draw on
    annotated_frame = frame.copy()
    
    prediction_text = "No hand detected"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. Get Bounding Box
            bbox = get_hand_bounding_box(hand_landmarks, frame.shape)
            
            # Draw the bounding box
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # 2. Crop the hand
            cropped_hand = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            if cropped_hand.size > 0:
                # 3. Preprocess the cropped image
                img_resized = cv2.resize(cropped_hand, (128, 128)) # Your model's input size
                img_array = np.asarray(img_resized)
                img_expanded = np.expand_dims(img_array, axis=0) # Shape: (1, 128, 128, 3)

                # 4. Predict
                # The Rescaling(1./255) layer is part of your saved model
                prediction = model.predict(img_expanded)
                
                # 5. Get the result
                pred_index = np.argmax(prediction)
                pred_label = labels[pred_index]
                confidence = np.max(prediction)
                
                prediction_text = f"{pred_label} ({confidence * 100:.1f}%)"
                
                # Display the prediction
                cv2.putText(annotated_frame, prediction_text, (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    frame_placeholder.image(annotated_frame, channels="BGR")
    
    # Display the top prediction in a separate area
    result_placeholder.markdown(f"**Prediction:** ## {prediction_text}")

# --- 6. Cleanup ---
cap.release()
cv2.destroyAllWindows()
st.write("Webcam stopped.")