/*
 * LYMPHA — ESP32 Sensor Firmware
 * ================================
 * Hardware:
 *   ESP32 WROOM-32 (OceanLabz, 30-pin)
 *   pH     : PH-4502C board   → GPIO34 (5V powered, ADC input only)
 *   Temp   : DS18B20           → GPIO4  (4.7kΩ pullup to 3.3V)
 *   Turb   : DFRobot Gravity   → GPIO32 (voltage divider: 10kΩ+18kΩ)
 *   TDS    : TDS Meter V1.0    → GPIO33 (3.3V powered)
 *
 * Sends JSON to POST /sensor/push every SEND_INTERVAL_MS milliseconds.
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ─── CONFIGURATION ───────────────────────────────────────
const char* WIFI_SSID     = "lenovo";
const char* WIFI_PASSWORD = "kiwukoora";
const char* SERVER_URL    = "http://192.168.137.1:8000/sensor/push";
// Find your PC IP: run `ipconfig` on Windows, use IPv4 address
// e.g. "http://192.168.1.105:8000/sensor/push"

const int SEND_INTERVAL_MS = 5000;   // send every 5 seconds

// ─── PIN DEFINITIONS ─────────────────────────────────────
#define PIN_TEMP     4
#define PIN_PH       34
#define PIN_TURB     32
#define PIN_TDS      33

// ─── pH CALIBRATION ──────────────────────────────────────
const float PH_NEUTRAL_VOLTAGE = 3.196;  // measured for this board
const int   PH_SAMPLES         = 50;

// ─── TDS CALIBRATION ─────────────────────────────────────
// If you have a reference TDS meter, dip both in the same water.
// Set TDS_SCALE = reference_reading / raw_reading.
// Example: reference=200ppm, raw=10ppm → TDS_SCALE = 20.0
const float TDS_SCALE = 15.0; // calibration: raw ~10ppm × 15 ≈ 150ppm for typical tap water

// ─── DS18B20 SETUP ───────────────────────────────────────
OneWire           oneWire(PIN_TEMP);
DallasTemperature tempSensor(&oneWire);

// ─── GLOBALS ─────────────────────────────────────────────
unsigned long lastSendMs = 0;

// ─────────────────────────────────────────────────────────
// SENSOR READING FUNCTIONS
// ─────────────────────────────────────────────────────────

float readTemperature() {
  tempSensor.requestTemperatures();
  float t = tempSensor.getTempCByIndex(0);
  if (t == DEVICE_DISCONNECTED_C) return -1.0;
  return t;
}

float readPH() {
  long sum = 0;
  for (int i = 0; i < PH_SAMPLES; i++) {
    sum += analogRead(PIN_PH);
    delay(10);
  }
  float raw     = sum / (float)PH_SAMPLES;
  float voltage = raw * (3.3 / 4095.0);   // ESP32 ADC reads 0–3.3V
  float ph      = 7.0 + ((PH_NEUTRAL_VOLTAGE - voltage) / 0.18);
  ph = constrain(ph, 0.0, 14.0);
  return ph;
}

// Returns turbidity in NTU.
// ── CALIBRATION ──────────────────────────────────────────
// Upload, open Serial Monitor, look for "[Turb raw=XXXX]" in clear water.
// Set TURB_CLEAR_RAW to that value, TURB_TURBID_RAW to the value in murky water.
// If clear water raw is LOW  → sensor outputs low-V for clear  (default below)
// If clear water raw is HIGH → flip: change the map() direction accordingly.
#define TURB_CLEAR_RAW   620    // raw ADC at clear/clean water (measured ~580 in tap water)
#define TURB_TURBID_RAW  3000   // raw ADC at very turbid water

float readTurbidity() {
  long sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += analogRead(PIN_TURB);
    delay(5);
  }
  int raw = (int)(sum / 10);
  Serial.printf("[Turb raw=%d]\n", raw);

  // Map raw → NTU: 0→CLEAR_RAW maps to 0–3 NTU (clear zone),
  //                 CLEAR_RAW→TURBID_RAW maps to 3–3000 NTU
  float ntu;
  if (raw <= TURB_CLEAR_RAW) {
    ntu = (float)raw / (float)TURB_CLEAR_RAW * 3.0f;
  } else if (raw <= TURB_TURBID_RAW) {
    ntu = 3.0f + (float)(raw - TURB_CLEAR_RAW) / (float)(TURB_TURBID_RAW - TURB_CLEAR_RAW) * 2997.0f;
  } else {
    ntu = 3000.0f;
  }
  return ntu;
}

float readTDS(float tempC) {
  float voltageSum = 0;
  for (int i = 0; i < 20; i++) {
    voltageSum += analogRead(PIN_TDS) * (3.3f / 4095.0f);
    delay(10);
  }
  float voltage = voltageSum / 20.0f;
  Serial.printf("[TDS raw_v=%.4fV]\n", voltage);

  // Temperature compensation: conductivity changes ~2% per °C from 25°C
  float usableTemp = (tempC > 0 && tempC < 100) ? tempC : 25.0f;
  float compensatedV = voltage / (1.0f + 0.02f * (usableTemp - 25.0f));

  float tds = (133.42f * pow(compensatedV, 3)
             - 255.86f * pow(compensatedV, 2)
             + 857.39f * compensatedV) * 0.5f;
  if (tds < 0) tds = 0;
  return tds * TDS_SCALE;
}

// ─────────────────────────────────────────────────────────
// WIFI
// ─────────────────────────────────────────────────────────

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.print("[WiFi] Connecting to ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected — IP: " + WiFi.localIP().toString());
  } else {
    Serial.println("\n[WiFi] Failed — will retry in loop");
  }
}

// ─────────────────────────────────────────────────────────
// HTTP POST
// ─────────────────────────────────────────────────────────

void sendReading(float ph, float temperature, float turbidity, float tds) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[HTTP] Not connected — skipping send");
    return;
  }

  StaticJsonDocument<128> doc;
  doc["ph"]          = round(ph * 100.0) / 100.0;
  doc["temperature"] = round(temperature * 10.0) / 10.0;
  doc["turbidity"]   = round(turbidity * 10.0) / 10.0;
  doc["tds"]         = round(tds * 10.0) / 10.0;

  String body;
  serializeJson(doc, body);

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(4000);
  int code = http.POST(body);

  if (code > 0) {
    Serial.printf("[HTTP] POST %d — %s\n", code, http.getString().c_str());
  } else {
    Serial.printf("[HTTP] Error: %s\n", http.errorToString(code).c_str());
  }
  http.end();
}

// ─────────────────────────────────────────────────────────
// SETUP & LOOP
// ─────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== LYMPHA Sensor Node ===");

  tempSensor.begin();
  analogReadResolution(12);   // 0–4095

  connectWiFi();
}

void loop() {
  // Reconnect if dropped
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Reconnecting...");
    connectWiFi();
  }

  unsigned long now = millis();
  if (now - lastSendMs >= SEND_INTERVAL_MS) {
    lastSendMs = now;

    float temperature = readTemperature();
    float ph          = readPH();
    float turbidity   = readTurbidity();
    float tds         = readTDS(temperature);

    Serial.printf("[Sensors] pH=%.2f  Temp=%.1f°C  Turb=%.1f NTU  TDS=%.1f ppm\n",
                  ph, temperature, turbidity, tds);

    sendReading(ph, temperature, turbidity, tds);
  }
}
