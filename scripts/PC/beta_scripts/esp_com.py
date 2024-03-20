
import config
import serial.tools.list_ports
import warnings
import time
import csv
from datetime import datetime, timedelta


if config.usb_com_port is None:
    serial_port_list = serial.tools.list_ports.comports()
    selected_port = sorted(serial_port_list)[0]
    print("Iniciando com serial")
    ser = serial.Serial(port=selected_port.name, baudrate=9600, timeout=1)
    ser.setRTS(True)
    time.sleep(0.3)
    ser.setRTS(False)
    time.sleep(0.3)

ser.flush()

while True:
    try:
        if ser.in_waiting > 0:
            print(ser.readline().decode())
    except KeyboardInterrupt:
        ser.setRTS(True)
        time.sleep(0.3)
        ser.setRTS(False)
        time.sleep(0.3)
        ser.close()



