# data_sources/real_hardware_source.py
import serial
import time

def get_telemetry_stream(port, baud_rate):
    """
    A generator that reads live telemetry from a serial port.
    It yields a new JSON string whenever one is received.
    """
    print(f"--- RUNNING IN REAL HARDWARE MODE (Port: {port}) ---")
    
    while True: # Loop forever to handle reconnects
        try:
            ser = serial.Serial(port, baud_rate, timeout=1)
            print(f"Successfully connected to {port}. Waiting for data...")
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        yield line
        except serial.SerialException as e:
            print(f"Warning: Serial port error: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Shutting down hardware reader.")
            break