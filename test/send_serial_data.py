import warnings
import serial.tools.list_ports

serial_port_list = serial.tools.list_ports.comports()
serial_port_list_size = len(serial_port_list)
if(serial_port_list_size == 0):
	raise ("Não foi detectado nenhuma comunicação serial compatível")
elif(serial_port_list_size > 1):
	warnings.warn("Foram encontradas mais de uma porta serial, o código exercutaa apenas com a primeira")

selected_port = sorted(serial_port_list)[0]
arduino = serial.Serial(port=selected_port.name, baudrate=115200, timeout=.1)
print(f"Porta {selected_port.name} conectada")
