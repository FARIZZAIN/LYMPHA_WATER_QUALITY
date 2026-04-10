# 💧 LYMPHA — AI Water Intelligence Platform

Real-time water quality monitoring with ML-powered potability prediction, fish habitability analysis, and automated alerts.

---

## 📁 Project Structure

```
lympha/
├── backend/               ← FastAPI Python server
│   ├── main.py            ← All API endpoints
│   ├── email_alert.py     ← Email notification
│   ├── whatsapp_alert.py  ← WhatsApp via Twilio
│   ├── serial_reader.py   ← ESP32 serial reader
│   ├── scaler.pkl         ← Your trained scaler
│   └── requirements.txt
└── frontend/
    └── index.html         ← Complete website (no build step!)
```

---

## 🚀 STEP-BY-STEP SETUP

### Step 1 — Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Start the backend

```bash
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000/docs to see all API endpoints.

### Step 3 — Open the website

Just open `frontend/index.html` in your browser. That's it — no build step needed.

The website auto-detects if the backend is running. If not, it runs in demo mode with simulated data.

---

## 🔌 Connecting Your ESP32

### Option A — USB Serial (local only)
In `backend/main.py`, uncomment this line:
```python
threading.Thread(target=_serial_loop, daemon=True).start()
```
Make sure your ESP32 is on the right COM port in `serial_reader.py`.

### Option B — WiFi (works when hosted online)
Add this to your Arduino sketch — it will POST readings to your backend every 30 seconds:

```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* ssid = "YOUR_WIFI";
const char* password = "YOUR_PASSWORD";
const char* serverURL = "http://YOUR_RENDER_URL/sensor/push";
// Replace with your Render URL after deployment

void sendToServer(float ph, float temp, float turb, float tds) {
  HTTPClient http;
  http.begin(serverURL);
  http.addHeader("Content-Type", "application/json");

  StaticJsonDocument<200> doc;
  doc["pH"] = ph;
  doc["Temperature"] = temp;
  doc["Turbidity"] = turb;
  doc["TDS"] = tds;

  String body;
  serializeJson(doc, body);
  int code = http.POST(body);
  http.end();
}

void loop() {
  // Read your sensors here...
  float ph = readPH();
  float temp = readTemp();
  float turb = readTurbidity();
  float tds = readTDS();

  sendToServer(ph, temp, turb, tds);
  delay(30000); // every 30 seconds
}
```

---

## 🌐 DEPLOYMENT

### Deploy Backend to Render (free)

1. Create a GitHub repo and push the `backend/` folder to it
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set these values:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port 10000`
   - **Environment:** Python 3.11
5. Click Deploy — Render gives you a URL like `https://lympha-api.onrender.com`

### Deploy Frontend to Netlify (free)

1. Go to https://netlify.com → Add new site → Deploy manually
2. Drag and drop the `frontend/` folder
3. Done — Netlify gives you a URL like `https://lympha.netlify.app`

### Connect Frontend to Backend

Open `frontend/index.html` and update line 1 of the script:
```javascript
const API_BASE = "https://lympha-api.onrender.com";  // your Render URL
```

---

## 🔮 Activating the Forecasting Model

When you're ready to wire up the GCN-STAE model:

1. Copy your `.pt` checkpoint file into `backend/`
2. In `backend/main.py`, replace the `/forecast/next` stub with:

```python
from live_infer import LiveForecaster

forecaster = LiveForecaster("your_checkpoint.pt")

@app.get("/forecast/next")
def get_forecast():
    reading = {**_latest_sensor}  # use latest sensor data
    prediction = forecaster.update_with_reading(reading)
    if prediction is None:
        return {"status": "warming_up", "message": "Need more data"}
    return {"status": "ok", "horizon_minutes": 10, "forecast": prediction}
```

3. The frontend forecast page will automatically display real predictions.

---

## 🔑 Environment Variables (for production)

Set these in Render's dashboard under Environment:

| Variable | Value |
|---|---|
| `TWILIO_ACCOUNT_SID` | Your Twilio SID |
| `TWILIO_AUTH_TOKEN` | Your Twilio token |

---

## 📧 Alert Configuration

Edit your email credentials in `backend/email_alert.py`:
```python
sender = "your_gmail@gmail.com"
app_password = "your_app_password"  # Google App Password, not your main password
```

To get a Gmail App Password: Google Account → Security → 2-Step Verification → App Passwords
