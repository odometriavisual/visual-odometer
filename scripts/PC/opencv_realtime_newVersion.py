#import sys
#sys.path.insert(0, r'C:\Users\Panther\PycharmProjects\virtualencoder')

import cv2
import serial.tools.list_ports

from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
from virtualencoder.visualodometry.svd_decomposition import optimized_svd_method
from virtualencoder.visualodometry.dsp_utils import cv2_to_nparray_grayscale, image_preprocessing

import config

import time
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

# --- definição de variáveis necessárias --- #

frame_num = -10
total_deltax = 0
total_deltay = 0

M = None
N = None
start_time = 0

resto_x = 0
resto_y = 0

intxacumulado = 0
intyacumulado = 0

gyroData = [0,0,0,0]

# --- Funções reservadas para envio e armazenamento de dados --- #

def checkSerialInput():
    global serial_giroscopio, gyroData
    if (serial_giroscopio.in_waiting > 0):
        ser_line = serial_giroscopio.readline().decode()
        gyroData = [float(x) for x in ser_line.split(",")]
        serial_giroscopio.flush()

def serialSendEncoder(x, y):
    global resto_x, resto_y, serial_encoder, intxacumulado, intyacumulado

    x += resto_x
    y += resto_y

    intx = int(x)
    inty = int(y)

    text_to_send = f"0,{intx}, {inty} \n"
    serial_encoder.write(text_to_send.encode())

    resto_x = x - intx
    resto_y = y - inty

# ----- Configuração de comunicação Serial ----- #

print("")
for port in ports:
    if port.serial_number == "5698028262":
        print("Iniciando conecção com o modulo do giroscópio")
        serial_giroscopio = serial.Serial(port=port.device, baudrate=9600, timeout=1)
        serial_giroscopio.setRTS(True)
        time.sleep(0.3)
        serial_giroscopio.setRTS(False)
        time.sleep(0.3)
    else:
        serial_encoder = serial.Serial(port=port.device, baudrate=115200, timeout=1)
        print("Iniciando comunicação com o modulo pulsador")
        time.sleep(0.6)

print("Testando comunicação serial: encoder")
_ = serial_encoder.read()
print("Testando comunicação serial: giroscópio server")
_ = serial_giroscopio.read()
print("Comunicação serial OK")
print("")

# -- Inicio da configuração da câmera --- #

print('Pegando acesso a camera, isso pode demorar um pouco...')
vid = cv2.VideoCapture(config.camera_id)

ret, frame = vid.read()
img_array = cv2_to_nparray_grayscale(frame)
if config.camera_exposure != None:
    print("Definindo exposição da câmera")
    vid.set(cv2.CAP_PROP_EXPOSURE, config.camera_exposure)


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

            serialSendEncoder(multiplied_deltax, multiplied_deltay) #<- Envia informações de deslocamento para o modulo pulsador

            total_deltax = total_deltax + multiplied_deltax
            total_deltay = total_deltay + multiplied_deltay

            # Exemplo de array para ser salvo:
            array_to_save = [time.time(),gyroData, deltax, deltay]
            print(array_to_save)
            # Recomendo limitar o salvamento a um intervalo. Mas é só uma sugestão mesmo.

        frame_num = frame_num + 1
        img_processed_old = img_processed

    except KeyboardInterrupt:

        # Fechando portas Seriais
        serial_giroscopio.setRTS(True)
        time.sleep(0.3)
        serial_giroscopio.setRTS(False)
        time.sleep(0.3)
        serial_giroscopio.close()
        serial_encoder.close()
        #-----------------------#

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
    except Exception as exc:
        print("Erro:", exc)
