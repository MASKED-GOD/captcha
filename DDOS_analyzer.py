import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# Load the dataset (replace with actual path to 'example_dataset.csv')
# This dataset should contain network traffic data with features like IP addresses, packet size, timestamp, etc.
data = pd.read_csv('example_dataset.csv')

# Feature Engineering
# We'll assume some basic columns in the dataset, like:
# 'src_ip', 'dst_ip', 'packet_size', 'timestamp', 'label' (label = 0 for normal traffic, 1 for DDoS attack)
# You may need to modify the feature engineering depending on the actual dataset structure.

def preprocess_data(data):
    """Process the dataset and extract relevant features for the ML model."""
    # Example features: Number of packets from the same source in a short time window
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Calculate the time difference between consecutive packets from the same source
    data['time_diff'] = data.groupby('src_ip')['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # Number of packets from the same source in the last 10 seconds (example window)
    data['packet_rate'] = data.groupby('src_ip').rolling('10s', on='timestamp').size().reset_index(0, drop=True)
    
    # Drop irrelevant columns for simplicity
    features = data[['packet_size', 'time_diff', 'packet_rate']]  # You can add more features if available
    
    return features, data['label']

# Preprocess the data
X, y = preprocess_data(data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
joblib.dump(model, 'ddos_model.pkl')
