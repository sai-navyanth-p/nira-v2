# NIRA (Network Incident Response Assistant) - v2
# FILE: traffic_simulator_v2.py
#
# This script simulates an external network tap.
# It reads the CSV file and sends each row as an HTTP POST
# request to our main_v2.py server.
#
# To run this, we must first install 'httpx':
# pip install httpx

import pandas as pd
import httpx
import time
import random
import json
import math

# --- 1. Configuration ---
BACKEND_API_URL = "http://127.0.0.1:8000/api/v2/traffic"
CSV_FILE_PATH = "UNSW_NB15_testing-set.csv"
PACKET_INTERVAL_MIN = 0.5  # Min seconds between packets
PACKET_INTERVAL_MAX = 2.0  # Max seconds between packets

# --- 2. Load Data ---
print("[NIRA v2 | Simulator] Loading simulation data...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
    # Convert dataframe to a list of dictionaries (like JSON objects)
    # 'orient=records' makes each row a dict
    traffic_data = df.to_dict(orient='records')
    print(f"[NIRA v2 | Simulator] Loaded {len(traffic_data)} packets.")
except FileNotFoundError:
    print(f"[FATAL] Could not find '{CSV_FILE_PATH}'.")
    print("Please make sure the file is in the same directory.")
    exit()
except Exception as e:
    print(f"[FATAL] Error loading CSV: {e}")
    exit()

def clean_packet(packet):
    cleaned = {}
    for key, value in packet.items():
        # Fix NaN
        if value is None or (isinstance(value, float) and math.isnan(value)):
            cleaned[key] = 0
        # Fix +inf / -inf
        elif isinstance(value, float) and (value == float("inf") or value == float("-inf")):
            cleaned[key] = 0
        else:
            cleaned[key] = value
    return cleaned


# --- 3. Run Simulation Loop ---
def run_simulation():
    print(f"[NIRA v2 | Simulator] Starting traffic simulation...")
    print(f"Targeting backend at: {BACKEND_API_URL}")
    
    with httpx.Client(timeout=10.0) as client:
        # Loop through the data indefinitely
        while True:
            try:
                # Get a random packet from our dataset
                packet = random.choice(traffic_data)

                packet = clean_packet(packet) 

                label_value = packet.get("label")
                print(f"Label of packet: {label_value}")
                
                response = client.post(BACKEND_API_URL, json=packet)
                
                if response.status_code == 200:
                    src_ip_log = packet.get('srcip') or "N/A"
                    print(f"Packet sent (srcip: {src_ip_log}) -> {response.json().get('status')}")
                else:
                    print(f"Error sending packet: {response.status_code} - {response.text}")
                
                # Wait for a random interval to simulate real traffic
                sleep_time = random.uniform(PACKET_INTERVAL_MIN, PACKET_INTERVAL_MAX)
                time.sleep(sleep_time)
                
            except httpx.ConnectError:
                print(f"Connection Error: Is the backend server ('main_v2.py') running?")
                print("Retrying in 5 seconds...")
                time.sleep(5)
            except KeyboardInterrupt:
                print("\n[NIRA v2 | Simulator] Simulation stopped by user.")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                time.sleep(2)

if __name__ == "__main__":
    run_simulation()