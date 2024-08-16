#5698028407
from serial.tools import list_ports

ports = list(list_ports.comports())

for port in ports:
    print(port.serial_number)
