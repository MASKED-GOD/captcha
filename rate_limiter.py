import time
from collections import defaultdict

# Configure rate limit settings
RATE_LIMIT = 5  # Number of allowed requests
TIME_WINDOW = 30  # Time window in seconds (e.g., 60 seconds)

# Store the request history for each user (IP address)
request_history = defaultdict(list)

def is_rate_limited(user_ip):
    """
    Check if a user is rate-limited based on their IP address.
    
    :param user_ip: IP address of the user making the request
    :return: True if the user is rate-limited, False otherwise
    """
    current_time = time.time()
    
    # Get the request timestamps for the user
    request_times = request_history[user_ip]
    
    # Remove requests that are outside the time window
    request_times = [t for t in request_times if current_time - t < TIME_WINDOW]
    request_history[user_ip] = request_times
    
    # Check if the number of requests exceeds the rate limit
    if len(request_times) >= RATE_LIMIT:
        return True
    else:
        # Add the current request timestamp
        request_history[user_ip].append(current_time)
        return False

def handle_request(user_ip):
    """
    Handle a request from the user and check if they are rate-limited.
    
    :param user_ip: IP address of the user making the request
    :return: None
    """
    if is_rate_limited(user_ip):
        print(f"User {user_ip} is rate-limited. Too many requests.")
    else:
        print(f"User {user_ip}'s request is processed.")

# Example usage
if __name__ == "__main__":
    # Simulate requests from users
    user_ip = "192.168.1.1"
    
    # Simulate 7 requests from the same user in quick succession
    for _ in range(7):
        handle_request(user_ip)
        time.sleep(10)  # Simulate 10-second intervals between requests
