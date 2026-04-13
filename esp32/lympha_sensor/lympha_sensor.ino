/*
 * LYMPHA — ESP32 Sensor Firmware
 * Hardware:
 *   pH     : PH-4502C board   → GPIO34
 *   Temp   : DS18B20           → GPIO4  (4.7kΩ pullup to 3.3V)
 *   Turb   : DFRobot Gravity   → GPIO32 (voltage divider: 10kΩ+18kΩ)
 *   TDS    : TDS Meter V1.0    → GPIO33 (3.3V powered)
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <OneWire.h>
#include <DallasTemperature.h>

const char* WIFI_SSID     = "lenovo";
const char* WIFI_PASSWORD = "kiwukoora";
const char* SERVER_URL    = "http://192.168.137.1:8000/sensor/push";

const int SEND_INTERVAL_MS = 5000;

#define PIN_TEMP     4
#define PIN_PH       34
#define PIN_TURB     32
#define PIN_TDS      33

const float PH_NEUTRAL_VOLTAGE = 3.196;  // measured at pH 7.0 for this board
const int   PH_SAMPLES         = 50;

// TDS_SCALE: set to (reference_ppm / raw_ppm) using a calibrated reference meter
const float TDS_SCALE = 15.0;

// Turbidity calibration — adjust based on Serial Monitor raw values:
//   TURB_CLEAR_RAW  : raw ADC in clean water  (measured ~580)
//   TURB_TURBID_RAW : raw ADC in turbid water
#define TURB_CLEAR_RAW   620
#define TURB_TURBID_RAW  3000

OneWire           oneWire(PIN_TEMP);
DallasTemperature tempSensor(&oneWire);

unsigned long lastSendMs = 0;


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
  float voltage = (sum / (float)PH_SAMPLES) * (3.3 / 4095.0);
  float ph = 7.0 + ((PH_NEUTRAL_VOLTAGE - voltage) / 0.18);
  return constrain(ph, 0.0, 14.0);
}

float readTurbidity() {
  long sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += analogRead(PIN_TURB);
    delay(5);
  }
  int raw = (int)(sum / 10);
  Serial.printf("[Turb raw=%d]\n", raw);

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

  float usableTemp   = (tempC > 0 && tempC < 100) ? tempC : 25.0f;
  float compensatedV = voltage / (1.0f + 0.02f * (usableTemp - 25.0f));

  float tds = (133.42f * pow(compensatedV, 3)
             - 255.86f * pow(compensatedV, 2)
             + 857.39f * compensatedV) * 0.5f;
  if (tds < 0) tds = 0;
  return tds * TDS_SCALE;
}


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


void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== LYMPHA Sensor Node ===");
  tempSensor.begin();
  analogReadResolution(12);
  connectWiFi();
}

void loop() {
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
