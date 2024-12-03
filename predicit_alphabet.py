import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import cv2
import os

# Load the dataset
dataset_path = r"D:\python\ML projects\air_canvas\Handwritten_Alphabets_Data.csv"
data = pd.read_csv(dataset_path, header=None)

# Preprocessing the data
X = data.iloc[:, 1:].values  # Features (784 columns)
y = data.iloc[:, 0].values   # Labels

# Normalize the data (scale pixel values to [0, 1])
X = X / 255.0

# Convert the labels to one-hot encoding
y_one_hot = pd.get_dummies(y).values

# Split the dataset into training (80%) and testing (20%)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y_one_hot[:train_size], y_one_hot[train_size:]

# Build the model
model = Sequential([
    Dense(512, input_shape=(784,), activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')  # 26 output classes (A-Z)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the model
model.save('letter_recognition_model.h5')

# Map class indices to letters (A-Z)
class_to_letter = {i: chr(65 + i) for i in range(26)}

def classify_submission(irregularity_score, confidence):
    HUMAN_THRESHOLD = 0.20
    BOT_THRESHOLD = 0.10

    if irregularity_score > HUMAN_THRESHOLD:
        return 'Human'
    elif irregularity_score < BOT_THRESHOLD:
        return 'Bot'
    elif confidence < 0.8:
        return 'Uncertain'
    else:
        return 'Uncertain'

def calculate_irregularity_score(img):
    """
    Calculate the irregularity score based on the contour analysis.
    """
    edges = cv2.Canny(img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    straight_line_score = 0
    total_contours = 0
    
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            
            # Calculate deviation from a perfect straight line
            deviation = np.std(np.diff(contour, axis=0))
            straight_line_score += deviation
            total_contours += 1
    
    if total_contours > 0:
        straight_line_score /= total_contours
    
    return straight_line_score

def evaluate_irregularity(image_path, model):
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (28, 28))
    img_flattened = img_resized.flatten().astype('float32') / 255.0
    img_flattened = np.expand_dims(img_flattened, axis=0)
    
    # Predict using the model
    predictions = model.predict(img_flattened)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_letter = class_to_letter[predicted_class]
    
    # Calculate confidence and irregularity score
    confidence = np.max(predictions)
    irregularity_score = calculate_irregularity_score(img_resized)
    
    # Classify as Human or Bot
    classification = classify_submission(irregularity_score, confidence)
    
    return predicted_letter, irregularity_score, classification, confidence

# Example usage
image_path = r"D:\python\ML projects\air_canvas\img_to_check\captcha_box.png"
try:
    predicted_letter, irregularity_score, classification, confidence = evaluate_irregularity(image_path, model)
    print(f"Predicted letter: {predicted_letter}")
    print(f"Irregularity score: {irregularity_score}")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence}")

except Exception as e:
    print(f"Error: {e}")

model.summary()
