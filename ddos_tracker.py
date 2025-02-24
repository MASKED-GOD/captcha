# ddos_tracker.py

import time
import uuid
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement

def log_attack(session, source_ip, destination_ip, attack_type):
    """
    🇫🇷 - Enregistre une attaque DDoS dans la base de données Cassandra.
    Arguments :
    - session : la session Cassandra pour exécuter les requêtes.
    - source_ip : l'adresse IP source de l'attaque.
    - destination_ip : l'adresse IP cible de l'attaque.
    - attack_type : le type d'attaque DDoS.
    
    🇬🇧 - Logs a DDoS attack into the Cassandra database.
    Arguments :
    - session : the Cassandra session to execute queries.
    - source_ip : the source IP address of the attack.
    - destination_ip : the target IP address of the attack.
    - attack_type : the type of DDoS attack.
    """
    try:
        query = """
        INSERT INTO attacks (id, timestamp, source_ip, destination_ip, attack_type)
        VALUES (%s, %s, %s, %s, %s)
        """
        # Create a simple statement with consistency level for fault tolerance
        statement = SimpleStatement(query, consistency_level=ConsistencyLevel.QUORUM)
        session.execute(statement, (uuid.uuid4(), time.time(), source_ip, destination_ip, attack_type))
        print(f"DDoS attack logged: {source_ip} -> {destination_ip}, Type: {attack_type}")
    except Exception as e:
        print(f"Error logging attack: {e}")

def main():
    """
    🇫🇷 - Fonction principale qui initialise la connexion à Cassandra et simule la réception de données d'attaque DDoS.
    🇬🇧 - Main function that initializes the connection to Cassandra and simulates receiving DDoS attack data.
    """
    try:
        # Connect to Cassandra cluster
        cluster = Cluster(['127.0.0.1'])  # Address of the Cassandra cluster
        session = cluster.connect('ddos')  # Connect to the 'ddos' keyspace
        
        print("DDoS Tracker started...")
        
        attack_types = ["SYN Flood", "UDP Flood", "HTTP GET Flood", "ICMP Flood"]
        
        while True:
            # Simulate receiving dynamic DDoS attack data
            source_ip = f"192.168.1.{uuid.uuid4().int % 255}"  # Randomize last part of IP
            destination_ip = f"10.0.0.{uuid.uuid4().int % 255}"  # Randomize destination IP
            attack_type = attack_types[uuid.uuid4().int % len(attack_types)]  # Random attack type
            
            log_attack(session, source_ip, destination_ip, attack_type)
            
            time.sleep(5)  # Wait for 5 seconds before the next attack
            
    except Exception as e:
        print(f"Error connecting to Cassandra: {e}")
    finally:
        # Ensure the cluster and session are properly closed
        if session:
            session.shutdown()
        if cluster:
            cluster.shutdown()

if __name__ == "__main__":
    main()
