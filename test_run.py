import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2

# Define constants
img_width, img_height = 28, 28

# Load the pre-trained model
model = keras.models.load_model('letter_recognition_model.h5')

def classify_submission(irregularity_score,confidence):
    HUMAN_THRESHOLD = 0.20
    BOT_THRESHOLD = 0.10

    print(f"Evaluating with irregularity score: {irregularity_score}")

    if irregularity_score > HUMAN_THRESHOLD:
        return 'Human'
    elif irregularity_score < BOT_THRESHOLD:
        return 'Bot'
    elif confidence < 0.8:
        return 'Uncertain'  # Add confidence as a factor
    else:
        return 'Uncertain'

def segment_characters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_images = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        char_img = img[y:y+h, x:x+w]
        char_images.append(char_img)

    return char_images

def preprocess_image(image, img_width, img_height):
    # Resize the image to the required input size
    resized_img = cv2.resize(image, (img_width, img_height))
    normalized_img = resized_img / 255.0  # Normalize pixel values to [0, 1]
    
    # Flatten the image directly after resizing
    img_array = normalized_img.flatten().reshape(1, -1)  # Flatten the image into a (1, 784) shape

    return img_array  # Return the flattened array


def evaluate_characters(character_images, model):
    results = []
    for char_img in character_images:
        img_array = preprocess_image(char_img, img_width, img_height)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        confidence = np.max(predictions)
        irregularity_score = 1 - confidence
        classification = classify_submission(irregularity_score)
        results.append((predicted_class, irregularity_score, classification, confidence))
    return results

# Example usage
image_path = r"D:\python\ML projects\air_canvas\img_to_check\captcha_box.png"
character_images = segment_characters(image_path)
results = evaluate_characters(character_images, model)

for i, (predicted_class, irregularity_score, classification, confidence) in enumerate(results):
    print(f"Character {i}:")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Irregularity score: {irregularity_score}")
    print(f"  Classification: {classification}")
    print(f"  Confidence: {confidence}")
