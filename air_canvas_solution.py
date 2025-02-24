
import mediapipe as mp
import cv2
import numpy as np
import os
import time
import tensorflow as tf  # For digit prediction
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import mnist
import pandas as pd
from tkinter import Tk, Label, Button
import random

# Load pre-trained digit recognition model
model = tf.keras.models.load_model("D:\python\ML projects\air_canvas\handwritten_digits.h5", compile=False)  # Ensure the model path is correct

# Constants
ml = 150
max_x, max_y = 250 + ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0, 0

# Define box region for CAPTCHA
captcha_box = {
    'x1': 100,  # Left coordinate of the box
    'y1': 100,  # Top coordinate of the box
    'x2': 540,  # Right coordinate of the box
    'y2': 300   # Bottom coordinate of the box
}

# Submit button coordinates (below the CAPTCHA box)
button_x1, button_y1 = 320, 320
button_x2, button_y2 = 470, 370

# Get tools function
def getTool(x):
    if x < 150 + ml:
        return "draw"
    else:
        return "erase"

def index_raised(yi, y9):
    return (y9 - yi) > 40

# Initialize Mediapipe Hands and Drawing utilities
hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Initialize white canvas and precision layer
canvas = np.ones((480, 640, 3), dtype="uint8") * 100  # White canvas
precision_layer = np.zeros_like(canvas)

# Load tools image
tools = cv2.imread("tools.png")
tools = tools.astype('uint8')

# Load MNIST dataset
(_, _), (test_images, test_labels) = mnist.load_data()

# Preprocess test images
test_images = test_images / 255.0  # Normalize
test_images = test_images.reshape(-1, 28, 28, 1)  # Reshape for model

# Generate predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Classification report
report = classification_report(test_labels, predicted_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Classification Report:")
print(report_df)

# Save classification report as a CSV file
report_df.to_csv("classification_report.csv", index=True)

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Visualize some example predictions
fig, axes = plt.subplots(5, 5, figsize=(12, 12))
axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.imshow(test_images[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"True: {test_labels[i]})
Pred: {predicted_labels[i]}", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()

def display_prediction_gui(predicted_digit, random_digit):
    root = Tk()
    root.title("Prediction Result")

    Label(root, text="The Prediction Results Are:", font=("Helvetica", 16)).pack(pady=10)
    Label(root, text=f"Predicted Digit: {predicted_digit}", font=("Helvetica", 24), fg="blue").pack(pady=10)
    Label(root, text=f"Random Digit: {random_digit}", font=("Helvetica", 24), fg="green").pack(pady=10)

    if predicted_digit == random_digit:
        result_text = "Match: The prediction matches the random digit!"
        result_color = "green"
    else:
        result_text = "No Match: The prediction does not match the random digit."
        result_color = "red"

    Label(root, text=result_text, font=("Helvetica", 18), fg=result_color).pack(pady=20)

    Button(root, text="Close", command=root.destroy, font=("Helvetica", 14)).pack(pady=10)

    root.mainloop()

# Generate a random digit before starting the camera
random_digit = random.randint(0, 9)
print(f"Random Digit: {random_digit}")

cap = cv2.VideoCapture(0)
while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)

    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
    op = hand_landmark.process(rgb)

    # Clear precision layer for each frame
    precision_layer[:] = 0

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)

            if x < max_x and y < max_y and x > ml:
                if time_init:
                    ctime = time.time()
                    time_init = False
                ptime = time.time()

                rad -= 1

                if (ptime - ctime) > 0.8:
                    curr_tool = getTool(x)
                    print("your current tool set to: ", curr_tool)
                    time_init = True
                    rad = 40

            else:
                time_init = True
                rad = 40

            if curr_tool == "draw":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.line(canvas, (prevx, prevy), (x, y), (0, 0, 0), thick)  # Drawing with black color
                    prevx, prevy = x, y
                else:
                    prevx = x
                    prevy = y

            elif curr_tool == "erase":
                xi, yi = int(i.landmark[12].x * 640), int(i.landmark[12].y * 480)
                y9 = int(i.landmark[9].y * 480)

                if index_raised(yi, y9):
                    cv2.circle(canvas, (x, y), 35, (55, 55, 55), -1)  # Erasing with white color

    # Draw the CAPTCHA box
    cv2.rectangle(canvas, (captcha_box['x1'], captcha_box['y1']), 
                           (captcha_box['x2'], captcha_box['y2']), 
                           (0, 0, 0), 2)  # Black border for the box

    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)
            
            # Draw precision triangle
            triangle_points = np.array([
                [x, y], [x - 10, y + 20], [x + 10, y + 20]
            ], np.int32)
            cv2.polylines(precision_layer, [triangle_points], isClosed=True, color=(255, 255, 255), thickness=2)

    # Overlay tools image
    canvas[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, canvas[:max_y, ml:max_x], 0.3, 0)

    # Combine the precision layer with the canvas
    combined_canvas = cv2.add(canvas, precision_layer)

    # Display current tool
    cv2.putText(combined_canvas, curr_tool, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display random digit on the canvas
    cv2.putText(combined_canvas, f"Random Digit: {random_digit}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw the submit button
    cv2.rectangle(combined_canvas, (button_x1, button_y1), (button_x2, button_y2), (220, 108, 54), -1)
    cv2.putText(combined_canvas, "Submit", (button_x1 + 10, button_y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Check for submit button click
    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            x, y = int(i.landmark[8].x * 640), int(i.landmark[8].y * 480)
            if button_x1 < x < button_x2 and button_y1 < y < button_y2:
                captcha_region = canvas[captcha_box['y1']:captcha_box['y2'], captcha_box['x1']:captcha_box['x2']]
                gray_captcha = cv2.cvtColor(captcha_region, cv2.COLOR_BGR2GRAY)
                resized_captcha = cv2.resize(gray_captcha, (28, 28))  # Resize to match MNIST input
                normalized_captcha = resized_captcha / 255.0
                input_data = normalized_captcha.reshape(1, 28, 28, 1)  # Reshape for the model
                
                # Predict digit
                prediction = model.predict(input_data)
                digit = np.argmax(prediction)
                print(f"Predicted Digit: {digit}")
                display_prediction_gui(digit, random_digit)

                # Exit on submission
                cv2.destroyAllWindows()
                cap.release()
                break

    # Show the canvas
    cv2.imshow("Drawing Canvas", combined_canvas)

    if cv2.waitKey(1) == 27:  # Exit on 'ESC'
        cv2.destroyAllWindows()
        cap.release()
        break
