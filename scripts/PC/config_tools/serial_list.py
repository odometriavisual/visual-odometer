# Dá uma lista das portas seriais disponíveis
import serial.tools.list_ports
ports = serial.tools.list_ports.comports()

for port in sorted(ports):
        print("{}".format(port.serial_number))