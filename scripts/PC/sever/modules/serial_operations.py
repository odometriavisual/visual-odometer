# serial_operations.py

global gyroData, serial_giroscopio, offset

def checkSerialInput(first_time=False):
    global gyroData, serial_giroscopio, offset
    if serial_giroscopio.in_waiting > 0:
        ser_line = serial_giroscopio.readline().decode()
        gyroData = [float(x) for x in ser_line.split(",")]
        gyroData = [gyroData[0], gyroData[1], gyroData[2], gyroData[3]]
        serial_giroscopio.read_all()
        # if first_time is True:
        #     quat_first = [quat[0], -quat[1], -quat[2], -quat[3]]
        #     offset = quaternion_multiply(glob_quat, quat_first)
        # = quaternion_multiply(offset, quat)

def serialSendEncoder(x, y):
    global resto_x, resto_y, serial_pulsador, intxacumulado, intyacumulado, serial_pulsador

    x += resto_x
    y += resto_y

    intx = int(x)
    inty = int(y)

    text_to_send = f"{intx},{inty},0\n"
    serial_pulsador.write(text_to_send.encode())

    resto_x = x - intx
    resto_y = y - inty
