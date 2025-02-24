from scapy.all import sniff
from collections import defaultdict
import time
import logging
import threading
import joblib
import pandas as pd

# Load the trained machine learning model
model = joblib.load('ddos_model.pkl')

# Track the number of packets sent by each IP
packet_count = defaultdict(int)
TIME_WINDOW = 60  # Reset packet counts every 60 seconds

# Function to extract features from network packets
def extract_features(packet):
    """Extract features from packet for ML model."""
    src_ip = packet["IP"].src
    packet_count[src_ip] += 1  # Increment packet count
    # Example: time difference and packet rate (using packet_count)
    # For real-time, we'll use a basic example based on packet count for now.
    packet_size = len(packet)
    time_diff = time.time()  # Timestamp when the packet is received
    packet_rate = packet_count[src_ip] / TIME_WINDOW  # Simple packet rate example
    
    return [packet_size, time_diff, packet_rate]

# Function to analyze packets using the trained ML model
def analyze_packet(packet):
    """Analyze captured packets using ML model."""
    if packet.haslayer("IP"):
        features = extract_features(packet)  # Extract relevant features for the model
        prediction = model.predict([features])  # Make prediction (1 = DDoS, 0 = Normal)
        
        src_ip = packet["IP"].src
        if prediction == 1:  # If the model detects DDoS
            logging.warning(f"Potential DDoS detected from {src_ip}. Packet count: {packet_count[src_ip]}")
            print(f"Potential DDoS attack detected from IP: {src_ip}. Packet count: {packet_count[src_ip]}")

# Function to reset packet counts after the time window
def reset_counts():
    """Reset packet counts every time window."""
    global packet_count
    while True:
        time.sleep(TIME_WINDOW)
        packet_count = defaultdict(int)  # Reset counts
        logging.info("Packet counts reset")

# Function to start sniffing packets
def start_sniffing(interface):
    """Start capturing network packets."""
    print(f"Starting packet capture on interface: {interface}")
    logging.info(f"Starting packet capture on interface: {interface}")
    sniff(iface=interface, prn=analyze_packet, store=0, use_pcap=True)

# Main section of the program
if __name__ == "__main__":
    # Start a thread to reset packet counts every TIME_WINDOW seconds
    reset_thread = threading.Thread(target=reset_counts, daemon=True)
    reset_thread.start()

    # Start sniffing on the desired network interface (replace 'Ethernet' with your interface)
    try:
        start_sniffing("Ethernet")  # Replace with your actual interface
    except Exception as e:
        logging.error(f"Error starting packet sniffing: {e}")
