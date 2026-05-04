import serial
import time

def send_id(port, baudrate, device_id):
    try:
        # Initialize serial connection
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2) # Wait for connection to stabilize
        
        # Format the string and add a newline character
        message = f"id:{device_id}\n"
        
        # Encode to bytes and send
        ser.write(message.encode('utf-8'))
        print(f"Sent: {message.strip()}")
        
        ser.close()
    except Exception as e:
        print(f"Error: {e}")

# Usage
send_id('COM3', 115200, "123456789") # Change 'COM3' to your actual port