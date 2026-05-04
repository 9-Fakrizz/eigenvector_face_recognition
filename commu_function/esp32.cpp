String currentID = "";

/**
 * Non-blocking check for Serial data.
 * Returns the ID only when a full message is completed with '\n'.
 */
String checkSerialNonBlocking() {
  static String buffer = ""; // Persists between function calls

  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      // Message is complete!
      String result = "";
      buffer.trim();
      
      if (buffer.startsWith("id:")) {
        result = buffer.substring(3);
      }
      
      buffer = ""; // Reset buffer for next message
      return result;
    } else {
      // Still receiving; add character to buffer
      buffer += c;
    }
  }
  
  return ""; // Nothing finished yet
}

void setup() {
  Serial.begin(115200);
  pinMode(2, OUTPUT); // Built-in LED for testing
}

void loop() {
  // 1. Check for ID (Non-blocking)
  String received = checkSerialNonBlocking();
  if (received != "") {
    currentID = received;
    Serial.print("New ID Stored: ");
    Serial.println(currentID);
  }

  // 2. Proof of non-blocking: Blink an LED
  // This LED will blink perfectly even while data is being received.
  digitalWrite(2, (millis() / 500) % 2); 
}