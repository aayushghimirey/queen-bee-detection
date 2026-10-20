#!/bin/bash
# ============================================================
# MSPB2 API End-to-End Curl Demonstration Script
# Tests user registration, hive creation, device provisioning,
# assignment, telemetry upload, history, and prediction.
# ============================================================

set -e
BASE_URL="http://localhost:8000"

echo "=== 1. Health Check ==="
curl -s -X GET "${BASE_URL}/health" | jq .

echo -e "\n=== 2. Register Beekeeper ==="
curl -s -X POST "${BASE_URL}/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "beekeeper@apiary.org",
    "password": "BeekeeperPassword123!",
    "full_name": "Elena Rostova"
  }' | jq . || true

echo -e "\n=== 3. Obtain JWT Token ==="
TOKEN_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "beekeeper@apiary.org",
    "password": "BeekeeperPassword123!"
  }')
ACCESS_TOKEN=$(echo $TOKEN_RESPONSE | jq -r .access_token)
echo "JWT Token obtained: ${ACCESS_TOKEN:0:20}..."

echo -e "\n=== 4. Create Hive ==="
curl -s -X POST "${BASE_URL}/api/v1/hives" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "HIVE-CURL-01",
    "name": "Clover Meadow Hive 1",
    "apiary_name": "Valley Meadow",
    "latitude": 45.4215,
    "longitude": -75.6972,
    "description": "Standard 10-frame Langstroth colony"
  }' | jq .

echo -e "\n=== 5. Register ESP32 IoT Device ==="
DEVICE_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v1/iot-devices" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "ESP32-CURL-NODE-01",
    "device_name": "Brood Acoustic Node #1",
    "hardware_version": "ESP32-WROOM-32D",
    "firmware_version": "v1.2.0"
  }')
DEVICE_KEY=$(echo $DEVICE_RESPONSE | jq -r .api_key)
echo "Registered Device API Key: ${DEVICE_KEY}"

echo -e "\n=== 6. Assign Device to Hive ==="
curl -s -X POST "${BASE_URL}/api/v1/hives/HIVE-CURL-01/assignments" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "ESP32-CURL-NODE-01"
  }' | jq .

echo -e "\n=== 7. ESP32 Sends Telemetry + Synthetic WAV Audio ==="
# Generate 1-second test WAV if python available
python3 -c "
import numpy as np, scipy.io.wavfile as wf
fs = 15625
t = np.linspace(0, 1.0, fs, endpoint=False)
sig = (0.5 * np.sin(2 * np.pi * 213.62 * t) * 32767).astype(np.int16)
wf.write('/tmp/test_burst.wav', fs, sig)
"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

curl -s -X POST "${BASE_URL}/api/v1/hives/HIVE-CURL-01/telemetry" \
  -H "X-Device-Id: ESP32-CURL-NODE-01" \
  -H "X-Device-Key: ${DEVICE_KEY}" \
  -F "reading_json={\"published_at\": \"${TIMESTAMP}\", \"temperature\": 34.6, \"humidity\": 54.5, \"audio_density\": 15.2, \"audio_density_ratio\": 0.85, \"density_variation\": 4.1}" \
  -F "audio=@/tmp/test_burst.wav;type=audio/wav" | jq .

echo -e "\n=== 8. Inspect Hive Telemetry History ==="
curl -s -X GET "${BASE_URL}/api/v1/hives/HIVE-CURL-01/history?page=1&page_size=5" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" | jq .

echo -e "\n=== 9. On-Demand Colony Strength Prediction ==="
curl -s -X GET "${BASE_URL}/api/v1/hives/HIVE-CURL-01/prediction" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}" | jq .

echo -e "\n=== Flow Complete ==="
