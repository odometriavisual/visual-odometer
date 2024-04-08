
import config
import serial.tools.list_ports
import warnings
import time
import csv
from datetime import datetime, timedelta

def exitCode():
    ser.setRTS(False)
    time.sleep(0.5)
    ser.setRTS(True)
    time.sleep(0.5)
    ser.setRTS(False)
    exit()

if config.usb_com_port is None:
    serial_port_list = serial.tools.list_ports.comports()
    port_list = sorted(serial_port_list)
    print("Iniciando com serial")
    ser = serial.Serial(port=port_list[0].device, baudrate=115200, timeout=1)
    port_list.device()

    ser.setRTS(False)
    time.sleep(0.5)
    ser.setRTS(True)
    time.sleep(0.5)
    ser.setRTS(False)
    ser.flush()

while True:
    try:
        print(ser.readline().decode())
    except KeyboardInterrupt:
        exitCode()
    except TypeError as e:
        print(e)
        exitCode()




