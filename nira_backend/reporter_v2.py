import os
import requests
import json

print("[NIRA v2 | Reporter] Module Loading (OpenAI API)...")

# --- 1. Configuration ---
openai_api_key = "" 


model_name = "gpt-4o-mini"
api_url = "https://api.openai.com/v1/chat/completions"

if openai_api_key == "YOUR_API_KEY_GOES_HERE":
    print("\n[ERROR] API key is not set in 'reporter_v2.py'.")
    print("Please edit this file and add your OpenAI API key.")
    openai_api_key = None  # Prevent further execution
else:
    print(f"[NIRA v2 | Reporter] Using model: {model_name}")
    print("[NIRA v2 | Reporter] Module Ready.")


# --- 2. Report Generation ---

def generate_incident_report_v2(attack_type, geo_info, full_data_row):
    """
    Uses the OpenAI API to write a comprehensive, multi-part incident report.
    
    Args:
        attack_type (str): The name of the detected attack (e.g., "DoS").
        geo_info (dict): Geolocation data.
        full_data_row (dict): The complete row of data from the CSV, as a dict.
    """
    if not openai_api_key:
        return "Error: OpenAI API key not set. Check reporter_v2.py."


    system_prompt = """
    You are a Tier 2 SOC Analyst. Your job is to write an incident report
    for a newly detected network threat.
    
    A detection system has flagged a packet with a specific "attack_cat"
    (attack category). I will provide you with:
    1.  The detected 'attack_type'.
    2.  The 'geo_info' (Source IP's location/ISP).
    3.  The 'full_data_row' (a JSON blob of all 40+ features
        for the flagged traffic).
    
    Your task is to return a clear, structured report with four parts:
    
    1.  **Executive Summary:** (1-2 sentences) What is this? What is the
        immediate impact? (e.g., "A 'DoS' attack from a server in China
        is attempting to flood our web service...")
    
    2.  **Threat Analysis:** (2-3 sentences) Explain what this attack is
        and what its goal is, based on the data. Is it trying to steal
        data, disrupt service, or scan for weaknesses?
    
    3.  **Key Indicators (Evidence):** (Bulleted list) From the
        'full_data_row', pull out the 3-5 most suspicious features that
        justify this classification. (e.g., "- `sbytes`: 5,000,000 (abnormally
        high)", "- `service`: 'none' (suspicious for this port)")
    
    4.  **Recommended Action:** (1-2 sentences) What is the immediate,
        specific next step? (e.g., "Block source IP 1.2.3.4 at the
        firewall," "Isolate the destination machine...")
    
    Format your response *only* as the 4-part report.
    Do not add any other greetings or sign-offs.
    """
    
    # Create the user prompt with all the data
    user_prompt = f"""
    A network threat has been detected. Please generate the incident report.
    
    **Detected Attack Type:**
    {attack_type}
    
    **Geo-IP Information:**
    {json.dumps(geo_info, indent=2)}
    
    **Full Data Row (JSON):**
    {json.dumps(full_data_row, indent=2)}
    """

    # --- Construct the API Payload ---
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": user_prompt.strip()
            }
        ],
        "temperature": 0.2, # Factual
        "max_tokens": 1024, # Allow for a longer, more detailed report
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    # --- Make the API Call ---
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=15)
        
        # Check for HTTP errors
        response.raise_for_status() 
        
        result = response.json()
        
        # Extract the text from the response
        report_text = result['choices'][0]['message']['content']
        return report_text.strip()

    except requests.exceptions.HTTPError as http_err:
        print(f"[NIRA v2 | Reporter] HTTP error occurred: {http_err}")
        try:
            print(f"API Error: {response.json()}")
        except:
            pass
        return "Error: Failed to generate report (HTTP error)."
    except Exception as e:
        print(f"[NIRA v2 | Reporter] API Error: {e}")
        return f"Error: Failed to generate report ({e.__class__.__name__})."


# --- 3. Self-Test ---
if __name__ == "__main__":
    print("\n--- Running Reporter v2 Self-Test (OpenAI API) ---")
    
    if openai_api_key:
        # A sample row of data (as a dictionary)
        test_row = {
            "srcip": "175.45.176.3", "sport": 33661, "dstip": "149.171.126.9",
            "dsport": 1024, "proto": "udp", "state": "CON", "dur": 0.036133,
            "sbytes": 528, "dbytes": 304, "sttl": 31, "dttl": 29, "sloss": 0,
            "dloss": 0, "service": "dns", "Sload": 116890.3047, "Dload": 67311.02344,
            "Spkts": 2, "Dpkts": 2, "swin": 0, "dwin": 0, "stcpb": 0, "dtcpb": 0,
            "smeansz": 264, "dmeansz": 152, "trans_depth": 0, "res_bdy_len": 0,
            "Sjit": 0.0, "Djit": 0.0, "Stime": 1421927414, "Ltime": 1421927414,
            "Sintpkt": 0.036133, "Dintpkt": 0.035935, "tcprtt": 0.0, "synack": 0.0,
            "ackdat": 0.0, "is_sm_ips_ports": 0, "ct_state_ttl": 0,
            "ct_flw_http_mthd": 0, "is_ftp_login": 0, "ct_ftp_cmd": 0,
            "ct_srv_src": 2, "ct_srv_dst": 2, "ct_dst_ltm": 1, "ct_src_ltm": 1,
            "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 2
        }
        test_geo = {'country': 'Australia', 'city': 'Sydney', 'isp': 'TestISP', 'lat': 0, 'lon': 0}
        
        print("Generating test report for a 'DoS' attack...")
        report = generate_incident_report_v2(
            attack_type="DoS", 
            geo_info=test_geo,
            full_data_row=test_row
        )
        print("\n--- Generated Report (v2) ---")
        print(report)
        print("-------------------------------")
    else:
        print("Self-test skipped. API key not provided.")