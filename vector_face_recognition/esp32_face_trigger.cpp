/**
 * ESP32 Face ID Trigger
 * ─────────────────────────────────────────────────────────────
 * Pin D35 (GPIO35) — input-only, no internal pull-up available
 *
 * Flow:
 *   IDLE       → sensor LOW for debounce → send "TRIGGER\n"
 *   WAITING    → receive "ID:xxxxxxxxx" / "TIMEOUT" / "NO_MATCH" / "NO_MODEL"
 *              → print result, enter COOLDOWN
 *   COOLDOWN   → 10 s → back to IDLE, send "READY\n"
 *
 * Serial baud: 115200
 */

#include <Arduino.h>

// ─── Config ──────────────────────────────────────────────────
static const int     SENSOR_PIN           = 35;       // GPIO35 (input-only)
static const uint32_t SENSOR_DEBOUNCE_MS  = 120;      // sensor must stay LOW this long
static const uint32_t RESPONSE_TIMEOUT_MS = 38000UL;  // 38 s > Python's 30 s window
static const uint32_t COOLDOWN_MS         = 10000UL;  // 10 s cooldown after any result
static const uint32_t SERIAL_BAUD         = 115200;

// ─── State machine ────────────────────────────────────────────
enum class State : uint8_t {
    IDLE,
    WAITING,
    COOLDOWN
};

static State    state           = State::IDLE;
static uint32_t stateEnteredAt  = 0;

// ─── Sensor glitch filter ─────────────────────────────────────
static int      prevSensorVal   = HIGH;
static uint32_t sensorLowSince  = 0;
static bool     sensorArmed     = true; // prevents re-trigger while still LOW

// ─── Helpers ─────────────────────────────────────────────────
static void enterState(State s) {
    state          = s;
    stateEnteredAt = millis();
}

static void handleResponse(const String& line) {
    if (line.startsWith("ID:")) {
        String uid = line.substring(3);
        uid.trim();
        Serial.print("[ACCESS GRANTED] ID: ");
        Serial.println(uid);
        // TODO: trigger relay / door / LED here if needed
    } else if (line == "TIMEOUT") {
        Serial.println("[ACCESS DENIED] Recognition timeout");
    } else if (line == "NO_MATCH") {
        Serial.println("[ACCESS DENIED] No matching face");
    } else if (line == "NO_MODEL") {
        Serial.println("[ERROR] Python: no face model loaded");
    } else {
        Serial.print("[UNKNOWN RESPONSE] ");
        Serial.println(line);
    }
    enterState(State::COOLDOWN);
}

// ─── Arduino entry points ─────────────────────────────────────
void setup() {
    Serial.begin(SERIAL_BAUD);
    pinMode(SENSOR_PIN, INPUT);   // GPIO35: input-only, no pull-up

    delay(500);
    Serial.println("[ESP32] Face ID Trigger ready");
    Serial.println("READY");
}

void loop() {
    const uint32_t now        = millis();
    const int      sensorVal  = digitalRead(SENSOR_PIN);

    switch (state) {

        // ── IDLE ─────────────────────────────────────────────
        case State::IDLE: {
            if (sensorVal == LOW) {
                if (prevSensorVal == HIGH) {
                    // Falling edge — start timing
                    sensorLowSince = now;
                    sensorArmed    = true;
                }
                if (sensorArmed && (now - sensorLowSince >= SENSOR_DEBOUNCE_MS)) {
                    // Confirmed: sensor has been LOW long enough
                    sensorArmed = false;   // don't fire again until it goes HIGH
                    Serial.println("TRIGGER");
                    Serial.println("[ESP32] Trigger sent — waiting for recognition result");
                    enterState(State::WAITING);
                }
            } else {
                // Sensor back to HIGH
                sensorArmed = true;
            }
            prevSensorVal = sensorVal;
            break;
        }

        // ── WAITING ──────────────────────────────────────────
        case State::WAITING: {
            // Hard timeout: Python didn't respond in time
            if (now - stateEnteredAt >= RESPONSE_TIMEOUT_MS) {
                Serial.println("[ESP32] Response timeout — no reply from Python");
                enterState(State::COOLDOWN);
                break;
            }

            // Read a line from Python
            if (Serial.available()) {
                String line = Serial.readStringUntil('\n');
                line.trim();
                if (line.length() > 0) {
                    handleResponse(line);
                }
            }
            break;
        }

        // ── COOLDOWN ─────────────────────────────────────────
        case State::COOLDOWN: {
            if (now - stateEnteredAt >= COOLDOWN_MS) {
                // Reset sensor state so a still-LOW pin doesn't immediately re-trigger
                prevSensorVal = HIGH;
                sensorArmed   = true;
                enterState(State::IDLE);
                Serial.println("[ESP32] Cooldown done");
                Serial.println("READY");
            }
            break;
        }
    }

    delay(10);
}
