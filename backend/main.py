import pickle
import random
import threading
import smtplib
import time
import numpy as np
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from fish_survival import predict_fish_survival
from forecast import ForecastModel
from src.config import Config as ForecastConfig
from data_replay import DatasetReplayer


ALERT_COOLDOWN_SECONDS = 3600
_last_alert_time: float = 0.0

EMAIL_SENDER   = "farizzain255@gmail.com"
EMAIL_PASSWORD = "crgl kufq wmsk rleu"
EMAIL_RECEIVERS = [
    "achsagracin@gmail.com",
    "afrahakim1234@gmail.com",
    "fathimas.nazar1234@gmail.com",
]

THRESHOLDS = {
    "pH":             {"min": 6.5,  "max": 8.5},
    "Turbidity":      {"min": None, "max": 5.0},
    "TDS":            {"min": None, "max": 600.0},
    "Conductivity":   {"min": None, "max": 750.0},
    "Hardness":       {"min": None, "max": 300.0},
    "Chloramines":    {"min": None, "max": 4.0},
    "Sulfate":        {"min": None, "max": 250.0},
    "Organic_carbon": {"min": None, "max": 4.0},
    "Trihalomethanes":{"min": None, "max": 80.0},
}

# Kaggle "Solids" column = groundwater TDS (5k–61k mg/L), incompatible with
# our sensor (0–1k ppm). TDS and Conductivity are simulated for model input;
# sensor values are shown on the dashboard only.
SIMULATED_RANGES = {
    "Hardness":        (60.0,   300.0),
    "TDS":             (5000.0, 20000.0),
    "Chloramines":     (0.5,    4.0),
    "Sulfate":         (150.0,  250.0),
    "Conductivity":    (200.0,  700.0),
    "Organic_carbon":  (1.0,    4.0),
    "Trihalomethanes": (17.0,   80.0),
}

FISH_DO_RANGE = (4.0, 12.0)

# Must match FEATURE_ORDER in train_potability_model.py
FEATURE_ORDER = [
    "pH", "Hardness", "TDS", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]


print("[LYMPHA] Loading potability model...")
with open("models/potability_scaler.pkl", "rb") as f:
    _scaler_bundle    = pickle.load(f)
potability_scaler  = _scaler_bundle["scaler"]
POTABILITY_THRESHOLD = 0.40
POTABILITY_MODEL_TYPE = _scaler_bundle.get("model_type", "nn")
FEATURE_ORDER = _scaler_bundle.get("feature_order", FEATURE_ORDER)

if POTABILITY_MODEL_TYPE == "nn":
    potability_model = load_model("models/potability_model.h5")
else:
    with open("models/potability_model.h5", "rb") as f:
        _bundle = pickle.load(f)
    potability_model = _bundle["model"]
print(f"[LYMPHA] Potability model ready — type={POTABILITY_MODEL_TYPE}, threshold={POTABILITY_THRESHOLD}")

print("[LYMPHA] Fish survival scorer ready.")

print("[LYMPHA] Loading forecast model...")
FORECAST_CKPT   = r"C:\Users\Faris\Desktop\lympha_main\checkpoint_rl.pt"
REPLAY_CSV_PATH = "data/synthetic_demo.csv"
REPLAY_INTERVAL = 3.0
try:
    forecaster = ForecastModel(FORECAST_CKPT, ForecastConfig())
    FORECAST_LOADED = True
except Exception as e:
    print(f"[LYMPHA] Forecast model failed to load: {e}")
    forecaster = None
    FORECAST_LOADED = False

latest_reading: dict = {}
_lock = threading.Lock()
_forecast_cache: dict = {}


def _on_replay_row(row):
    global _forecast_cache
    if FORECAST_LOADED:
        forecaster.push_row(row)
        _forecast_cache = _build_forecast_response()


app = FastAPI(title="LYMPHA Water Quality API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ESP32Push(BaseModel):
    ph:          float
    temperature: float
    turbidity:   float
    tds:         float


class ManualReading(BaseModel):
    ph:              float = 7.0
    temperature:     float = 25.0
    turbidity:       float = 3.0
    tds:             float = 300.0
    conductivity:    float | None = None
    hardness:        float | None = None
    chloramines:     float | None = None
    sulfate:         float | None = None
    organic_carbon:  float | None = None
    trihalomethanes: float | None = None


def simulated_params() -> dict:
    # Fixed values near the Kaggle potable-class means
    return {
        "Hardness":        180.0,
        "TDS":             19000.0,
        "Chloramines":     3.5,
        "Sulfate":         220.0,
        "Conductivity":    420.0,
        "Organic_carbon":  2.5,
        "Trihalomethanes": 50.0,
    }


def find_issues(data: dict) -> list:
    issues = []
    for key, limits in THRESHOLDS.items():
        val = data.get(key)
        if val is None:
            continue
        if limits["min"] is not None and val < limits["min"]:
            issues.append({"parameter": key, "value": val,
                           "reason": f"below minimum ({limits['min']})"})
        if limits["max"] is not None and val > limits["max"]:
            issues.append({"parameter": key, "value": val,
                           "reason": f"above maximum ({limits['max']})"})
    return issues


def run_model(live: dict, sim: dict, send_alerts: bool = True) -> dict:
    sensor_tds  = live.get("TDS", 0)
    sensor_cond = live.get("Conductivity", 0)

    model_data = {**live, **sim}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    model_data["pH_x_Chloramines"]      = model_data["pH"] * model_data["Chloramines"]
    model_data["TDS_div_Conductivity"]  = model_data["TDS"] / (model_data["Conductivity"] + 1e-6)
    model_data["Hardness_x_Sulfate"]    = model_data["Hardness"] * model_data["Sulfate"]
    model_data["Organic_x_Trihalometh"] = model_data["Organic_carbon"] * model_data["Trihalomethanes"]
    model_data["pH_squared"]            = model_data["pH"] ** 2
    model_data["Turbidity_x_TDS"]       = model_data["Turbidity"] * model_data["TDS"]

    features = np.array([[model_data[k] for k in FEATURE_ORDER]])
    features_scaled = potability_scaler.transform(features)
    if POTABILITY_MODEL_TYPE == "nn":
        probability = float(potability_model.predict(features_scaled, verbose=0)[0][0])
    else:
        probability = float(potability_model.predict_proba(features_scaled)[0][1])

    is_safe = probability >= POTABILITY_THRESHOLD

    # Rule-based checks use sensor-scale values (not Kaggle-scale)
    display_data = {**model_data, "TDS": sensor_tds, "Conductivity": sensor_cond}
    issues = find_issues(display_data)

    if live.get("Turbidity", 0) > 3:
        is_safe = False
        turb_display = round(live.get("TurbidityDisplay", live.get("Turbidity", 0)), 1)
        issues.append({
            "parameter": "Turbidity",
            "value": turb_display,
            "reason": "high turbidity detected"
        })

    result = {
        "timestamp":   timestamp,
        "safe":        is_safe,
        "probability": round(probability, 4),
        "label":       "SAFE" if is_safe else "NOT SAFE",
        "issues":      issues,
        "live_sensors": live,
        "simulated_sensors": sim,
        "readings": model_data,
        "data": {
            "pH":              live["pH"],
            "Temperature":     live.get("Temperature"),
            "Turbidity":       live.get("TurbidityDisplay", live.get("Turbidity")),
            "TDS":             sensor_tds,
            "Conductivity":    sensor_cond,
            "Hardness":        sim["Hardness"],
            "Chloramines":     sim["Chloramines"],
            "Sulfate":         sim["Sulfate"],
            "Organic_carbon":  sim["Organic_carbon"],
            "Trihalomethanes": sim["Trihalomethanes"],
        }
    }

    if not is_safe and issues and send_alerts:
        global _last_alert_time
        now_ts = time.time()
        if now_ts - _last_alert_time >= ALERT_COOLDOWN_SECONDS:
            _last_alert_time = now_ts
            threading.Thread(
                target=send_alert_email, args=(issues, timestamp), daemon=True
            ).start()

    print(f"[{timestamp}] {result['label']} (prob={probability:.3f}) "
          f"pH={live['pH']} Temp={live['Temperature']} "
          f"Turb={live['Turbidity']} TDS={live['TDS']}")
    return result


def send_alert_email(issues: list, timestamp: str):
    try:
        lines = "\n".join(
            f"  - {i['parameter']}: {i['value']} ({i['reason']})" for i in issues
        )
        body = (
            f"WATER QUALITY ALERT\n"
            f"Station : WQ-STATION-01\n"
            f"Time    : {timestamp}\n\n"
            f"Parameters outside safe limits:\n{lines}\n\n"
            f"Please check the water source immediately."
        )
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = ", ".join(EMAIL_RECEIVERS)
        msg["Subject"] = "Water Quality Alert — Unsafe Conditions Detected"
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVERS, msg.as_string())
        print(f"[Alert] Email sent at {timestamp}")
    except Exception as e:
        print(f"[Alert] Email failed: {e}")


try:
    replayer = DatasetReplayer(REPLAY_CSV_PATH, interval=REPLAY_INTERVAL)
    replayer.start(_on_replay_row)
    print(f"[LYMPHA] Dataset replay started ({REPLAY_INTERVAL}s interval).")
except Exception as e:
    print(f"[LYMPHA] Replay failed to start: {e}")
    replayer = None


@app.get("/health")
def health():
    with _lock:
        has_data = bool(latest_reading)
        last_ts  = latest_reading.get("timestamp", "never")
    return {"status": "ok", "has_reading": has_data, "last_received": last_ts}


@app.post("/sensor/push")
def sensor_push(reading: ESP32Push):
    global latest_reading

    turb_display = reading.turbidity
    if turb_display <= 3.0:
        turb_model = 2.0
    elif turb_display >= 3000:
        turb_model = 6.74
    else:
        # Map sensor NTU range (3–3000) → Kaggle Turbidity range (2.0–6.74)
        turb_model = 2.0 + ((turb_display - 3.0) / (3000.0 - 3.0)) * (6.74 - 2.0)

    live = {
        "pH":               reading.ph,
        "Temperature":      reading.temperature,
        "Turbidity":        turb_model,
        "TurbidityDisplay": turb_display,
        "TDS":              reading.tds,
        "Conductivity":     reading.tds / 0.64,
    }
    sim = simulated_params()
    result = run_model(live, sim)

    with _lock:
        latest_reading = result

    return {"status": "ok", "label": result["label"], "probability": result["probability"]}


@app.get("/sensor/live")
@app.get("/latest")
def sensor_live():
    with _lock:
        reading = dict(latest_reading)

    if not reading:
        return {
            "message": "No data yet — waiting for ESP32 to connect.",
            "data": None,
        }
    return reading


@app.post("/predict")
@app.post("/predict/potability")
def predict(reading: ManualReading):
    live = {
        "pH":          reading.ph,
        "Temperature": reading.temperature,
        "Turbidity":   reading.turbidity,
        "TDS":         reading.tds,
    }
    sim = simulated_params()
    live["Conductivity"] = reading.conductivity if reading.conductivity is not None else reading.tds / 0.64
    if reading.hardness        is not None: sim["Hardness"]        = reading.hardness
    if reading.chloramines     is not None: sim["Chloramines"]     = reading.chloramines
    if reading.sulfate         is not None: sim["Sulfate"]         = reading.sulfate
    if reading.organic_carbon  is not None: sim["Organic_carbon"]  = reading.organic_carbon
    if reading.trihalomethanes is not None: sim["Trihalomethanes"] = reading.trihalomethanes

    # Alerts only from live ESP32 readings, not the manual simulator
    result = run_model(live, sim, send_alerts=False)

    if result.get("issues"):
        result["safe"] = False
        result["label"] = "NOT SAFE"

    # Do not overwrite latest_reading — simulator must not pollute live dashboard

    prob = result["probability"]
    is_safe = result["safe"]
    confidence = prob if is_safe else (1.0 - prob)
    risk = "Low" if prob >= 0.65 else "Medium" if prob >= 0.4 else "High"
    return {
        **result,
        "potable":    is_safe,
        "confidence": round(confidence * 100, 1),
        "risk_level": risk,
    }


@app.post("/predict/fish")
def predict_fish(reading: ManualReading):
    water = {
        "pH":          reading.ph,
        "Temperature": reading.temperature,
        "Turbidity":   reading.turbidity,
        "DO":          round(random.uniform(*FISH_DO_RANGE), 2),
        "Conductivity": reading.conductivity,
    }
    water = {k: v for k, v in water.items() if v is not None}
    predictions = predict_fish_survival(water)
    return {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input":        water,
        "predictions":  predictions,
        "best_species": predictions[0]["species"] if predictions else None,
    }


def _build_forecast_response() -> dict:
    fill   = forecaster.buffer_fill
    needed = 48

    if fill < needed:
        return {
            "ready":       False,
            "buffer_fill": fill,
            "needed":      needed,
            "message":     f"Warming up — {fill}/{needed} readings collected ({needed - fill} remaining).",
        }

    preds = forecaster.predict()
    if preds is None:
        return {"ready": False, "buffer_fill": fill, "needed": needed}

    stations = {}
    for node, val in preds.items():
        param, st = node.rsplit("_", 1)
        stations.setdefault(st, {})[param] = val

    # Node layout: [Temp, pH, Cond, Turb] × 3 stations, offset by 4 per station
    PARAMS = {"Temp": 0, "pH": 1, "Cond": 2, "Turb": 3}

    eval_series_by_station = {}
    history_by_station = {}
    for st in range(3):
        offset = st * 4
        st_key = str(st + 1)
        eval_series_by_station[st_key] = {}
        history_by_station[st_key] = {}
        for p, base_idx in PARAMS.items():
            idx = base_idx + offset
            actual, predicted = forecaster.get_eval_series(idx)
            eval_series_by_station[st_key][p] = {"actual": actual, "predicted": predicted}
            history_by_station[st_key][p] = forecaster.get_history(idx)

    return {
        "ready":       True,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predictions": preds,
        "by_station":  stations,
        "eval_series": eval_series_by_station,
        "history":     history_by_station,
        "buffer_fill": fill,
    }


@app.get("/forecast/next")
def forecast_next():
    if not FORECAST_LOADED:
        return {"ready": False, "error": "Forecast model not loaded."}
    if not _forecast_cache:
        fill = forecaster.buffer_fill if forecaster else 0
        return {"ready": False, "buffer_fill": fill, "needed": 48,
                "message": f"Warming up — {fill}/48 readings collected ({48 - fill} remaining)."}
    return _forecast_cache


@app.post("/alerts/send")
def send_manual_alert(payload: dict):
    issues = payload.get("unsafe_parameters", [])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    threading.Thread(target=send_alert_email, args=(issues, timestamp), daemon=True).start()
    return {"status": "sent", "timestamp": timestamp}


@app.post("/settings/alerts")
def update_alert_settings(payload: dict):
    global EMAIL_RECEIVERS
    emails = payload.get("email_receivers", [])
    if not emails:
        return {"status": "error", "message": "No emails provided"}
    EMAIL_RECEIVERS = [e.strip() for e in emails if e.strip()]
    print(f"[Alert] Recipients updated: {EMAIL_RECEIVERS}")
    return {"status": "ok", "email_receivers": EMAIL_RECEIVERS}
