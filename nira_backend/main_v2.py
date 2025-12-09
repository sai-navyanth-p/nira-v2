import uvicorn
import asyncio
import json
import uuid
import requests
import pandas as pd
import joblib
import xgboost as xgb
from datetime import datetime
from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

try:
    from reporter_v2 import generate_incident_report_v2
except ImportError:
    print("[FATAL] reporter_v2.py not found.")
    exit()

# --- 1. Load AI Artifacts ---
print("[NIRA v2 | Server] Loading AI artifacts...")
try:
    model = joblib.load("nira_ensemble_model.joblib")
    
    preprocessor = joblib.load("nira_preprocessor.joblib")
    label_encoder = joblib.load("nira_label_encoder.joblib")
    feature_lists = joblib.load("nira_feature_lists.joblib")
    
    cat_features = feature_lists['cat_features']
    num_features = feature_lists['num_features']
    
    print("[NIRA v2 | Server] Ensemble AI loaded successfully.")
except FileNotFoundError as e:
    print(f"[FATAL] Missing artifact file: {e.filename}")
    print("Did you run NIRA_Advanced_Ensemble.py and download the files?")
    exit()

# --- 2. FastAPI App Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- 3. WebSocket Connection Manager ---
active_connections: list[WebSocket] = []

async def broadcast(message: str):
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except WebSocketDisconnect:
            active_connections.remove(connection)
        except Exception:
            active_connections.remove(connection)

# --- 4. WebSocket Endpoint ---
@app.websocket("/ws/v2")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    print("[NIRA v2 | Server] WebSocket connection opened.")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print("[NIRA v2 | Server] WebSocket connection closed.")
        
# --- 5. Geo-IP Function ---
def get_geolocation(ip):
    geo_info = {'lat': 0, 'lon': 0, 'city': 'N/A', 'country': 'N/A', 'isp': 'N/A'}
    
    if not ip or pd.isna(ip) or ip == "N/A" or ip.startswith(('10.', '192.168.', '172.16.')):
        geo_info['country'] = 'Private/Internal'
        return geo_info
        
    try:
        response = requests.get(f"http://ip-api.com/json/{ip}?fields=status,country,city,isp,lat,lon", timeout=1)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                geo_info.update(data)
                print(f"[NIRA v2 | Geo-IP] Location: {geo_info.get('city')}, {geo_info.get('country')}")
    except Exception as e:
        print(f"[NIRA v2 | Geo-IP] API Error: {e}")
        
    return geo_info

# --- 6. The Core "Brain" Function ---
async def process_traffic_packet(row_dict: dict):
    
    # 1. Broadcast packet receipt
    await broadcast(json.dumps({"type": "TRAFFIC_PACKET"}))
    
    # --- 2. DETECT (The Ensemble Brain) ---
    raw_data = pd.DataFrame([row_dict])

    X_live = raw_data.drop(columns=['attack_cat', 'id', 'label'], errors='ignore')
    
    # Clean data
    X_live[num_features] = X_live[num_features].fillna(0)
    X_live[cat_features] = X_live[cat_features].fillna('unknown')
    X_live['service'] = X_live['service'].replace('-', 'none')
    
    # Preprocess
    X_processed = preprocessor.transform(X_live)
    
    prediction_encoded = model.predict(X_processed)
    attack_type = label_encoder.inverse_transform(prediction_encoded)[0]

    # --- 3. DECIDE & REPORT ---
    if attack_type != 'Normal':
        print(f"\n[!!! NIRA v2] ATTACK DETECTED: {attack_type}")
        
        src_ip = row_dict.get('srcip', 'N/A')
        dest_port = row_dict.get('dsport', 'N/A')
        packet_count = row_dict.get('spkts', 0) + row_dict.get('dpkts', 0)
        
        geo_info = get_geolocation(src_ip)

        print("[NIRA v2 | Reporter] Generating incident report...")
        report_text = generate_incident_report_v2(
            attack_type, geo_info, row_dict
        )
        print(f"[NIRA v2 | Reporter] Report generated.")

        alert_payload = {
            'type': 'NEW_ALERT',
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'attackType': attack_type,
            'ip': src_ip,
            'port': dest_port,
            'packets': int(packet_count),
            'geo': geo_info,
            'report': report_text
        }
        
        await broadcast(json.dumps(alert_payload))

# --- 7. HTTP Endpoint ---
@app.post("/api/v2/traffic")
async def receive_traffic(request: Request):
    try:
        row_data = await request.json()
        asyncio.create_task(process_traffic_packet(row_data))
        return {"status": "packet received and processing"}
    except Exception as e:
        print(f"[NIRA v2 | Server] Error processing packet: {e}")
        return {"status": "error", "detail": str(e)}

# --- 8. Root Endpoint ---
@app.get("/")
def read_root():
    return {"NIRA": "Network Incident Response Assistant", "Version": "2.1 (Ensemble)"}

# --- 9. Run the Server ---
if __name__ == "__main__":
    print("[NIRA v2 | Server] Starting Uvicorn server at http://127.0.0.1:8000")
    uvicorn.run("main_v2:app", host="127.0.0.1", port=8000, reload=True)