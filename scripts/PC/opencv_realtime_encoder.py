import sys
import time
import warnings

sys.path.insert(0, r'C:\Users\Panther\PycharmProjects\virtualencoder')

w = "..\pasta"
import cv2
import serial.tools.list_ports

from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
from virtualencoder.visualodometry.svd_decomposition import optimized_svd_method
from virtualencoder.visualodometry.dsp_utils import cv2_to_nparray_grayscale, image_preprocessing

import config


resto_x = 0
resto_y = 0

intxacumulado = 0
intyacumulado = 0

gyroData = [0,0,0,0]
def atualizarPos(x,y):
    global resto_x, resto_y, serial_pulsador, intxacumulado, intyacumulado

    x += resto_x
    y += resto_y

    intx = int(x)
    inty = int(y)

    text_to_send = f"{intx},{inty},0\r\n"
    serial_pulsador.write(text_to_send.encode())
    #_ = arduino.readline()

    resto_x = x - intx
    resto_y = y - inty

def checkSerialInput():
    global gyroData, serial_giroscopio
    while (serial_giroscopio.in_waiting > 0):
        ser_line = serial_giroscopio.readline().decode()
        gyroData = [float(x) for x in ser_line.split(",")] #list  quaternium to biallhole
        serial_giroscopio.flush()

def endSerialComunication():
    serial_giroscopio.setRTS(False)
    time.sleep(0.3)
    serial_giroscopio.setRTS(True)
    time.sleep(0.3)
    serial_giroscopio.setRTS(False)
    time.sleep(0.3)
    serial_pulsador.setRTS(False)
    time.sleep(0.3)
    serial_pulsador.setRTS(True)
    time.sleep(0.3)
    serial_pulsador.setRTS(False)

ports = serial.tools.list_ports.comports()
for port in ports:
    print(port)
    if port.serial_number == "56CA000930":
        print("Iniciando conecção com o modulo do giroscópio")
        serial_giroscopio = serial.Serial(port=port.device, baudrate=115200, timeout=1)
        serial_giroscopio.setRTS(False)
        time.sleep(0.3)
        serial_giroscopio.setRTS(True)
        time.sleep(0.3)
        serial_giroscopio.setRTS(False)
        time.sleep(0.3)
    elif port.serial_number == "5698010135":
        print("Iniciando comunicação com o modulo pulsador")
        serial_pulsador = serial.Serial(port=port.device, baudrate=115200, timeout=1)
        serial_pulsador.setRTS(False)
        time.sleep(0.3)
        serial_pulsador.setRTS(True)
        time.sleep(0.3)
        serial_pulsador.setRTS(False)
        time.sleep(0.3)


# - Fim do setup Serial - #

print('Pegando acesso a camera, isso pode demorar um pouco...')
vid = cv2.VideoCapture(config.camera_id)
try:
    ret, frame = vid.read()
    img_array = cv2_to_nparray_grayscale(frame)
except:
    print("Não foi possível conectar a câmera, altere o id da camera no config.py")
    exit()
if config.camera_exposure != None:
    print("Definindo exposição da câmera")
    vid.set(cv2.CAP_PROP_EXPOSURE, config.camera_exposure)

frame_num = -10 #Definido para negativo, o frame será contabilizado apenas após a décima imagem
total_deltax = 0
total_deltay = 0

M = None
N = None
start_time = 0

while True:
    try:
        checkSerialInput()
        ret, frame = vid.read()
        img_array = cv2_to_nparray_grayscale(frame)
        img_windowed = apply_border_windowing_on_image(img_array, config.border_windowing_method)
        img_processed = image_preprocessing(img_array)
        if frame_num > 0:
            if (M == None):
                print("Script iniciado")
                start_time = time.time()
                M, N = img_array.shape
            deltax, deltay = optimized_svd_method(img_processed, img_processed_old, M, N, phase_windowing=config.phase_windowing)
            multiplied_deltax = config.deltax_multiplier * deltax
            multiplied_deltay = config.deltay_multiplier * deltay
            atualizarPos(multiplied_deltax,multiplied_deltay)
            total_deltax = total_deltax + (multiplied_deltax)
            total_deltay = total_deltay + (multiplied_deltay)

            print(f"Frame:  {frame_num:>3.2f}, delta:[{deltax:>5.2f},{deltay:>5.2f}], total_delta:[{total_deltax:>5.2f}, {total_deltay:>5.2f}, {gyroData}]")

        frame_num = frame_num + 1
        img_processed_old = img_processed
    except:
        endSerialComunication()
        vid.release()

        passed_time = (time.time() - start_time)
        print("--- %s seconds ---" % passed_time)
        print("--- %s  frames ---" % frame_num)

        fps = frame_num / passed_time
        print("--- %s     fps ---" % fps)
        print("")
        print(f"Total deltax: {total_deltax}")
        print(f"Total deltay: {total_deltay}")

        exit()

