import time
import warnings
import serial
import serial.tools.list_ports

usb_com_port = None


# [Posição desejada, Velocidade desejada]
sequence = [[200,100],
            [600,200],
            [1200,300],
            [2000,400],
            [4000,700],
            [0,700]
            ]



sequence_size = len(sequence)
contador = -1
first_time = True

if usb_com_port is None:
    print("Iniciando setup automático de comunicação Serial")
    serial_port_list = serial.tools.list_ports.comports()
    serial_port_list_size = len(serial_port_list)
    if (serial_port_list_size == 0):
        print("Não foi detectado nenhuma comunicação serial compatível")
        print("verifique se o módulo pulsador (arduino) está conectado")
        exit()
    elif (serial_port_list_size > 1):
        warnings.warn("ATENÇÃO - Foram encontradas mais de uma porta serial, o código exercutaa apenas com uma delas")
    selected_port = sorted(serial_port_list)[0]
    arduino_atuador = serial.Serial(port=selected_port.name, baudrate=115200, timeout=100000)
    print(f"Porta {selected_port.name} conectada")
    time.sleep(1)

else:
    try:
        arduino_atuador = serial.Serial(port=usb_com_port, baudrate=115200, timeout=100000)
    except:
        print("Erro na conexão da comunicação serial, é recomendado alterar a variável usb_com_port no config.py para None")
        exit()

while True:
    try:
        if (arduino_atuador.inWaiting() > 2):
            arduino_atuador.flushInput()
            contador = (contador + 1) % sequence_size
            position = sequence[contador][0]
            speed = sequence[contador][1]
            text_to_send = f"{position}, {speed}\n"
            print(text_to_send)
            arduino_atuador.write(text_to_send.encode())
            first_time = False
    except KeyboardInterrupt:
        arduino_atuador.close()
        exit()
