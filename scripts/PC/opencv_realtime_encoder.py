import sys
import time
import warnings

sys.path.insert(0, r'C:\Users\Panther\PycharmProjects\virtualencoder')

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

def atualizarPos(x,y):
    global resto_x, resto_y, arduino, intxacumulado, intyacumulado

    x += resto_x
    y += resto_y

    intx = int(x)
    inty = int(y)

    text_to_send = f"0,{intx}, {inty} \n"
    arduino.write(text_to_send.encode())
    #_ = arduino.readline()

    resto_x = x - intx
    resto_y = y - inty

if config.usb_com_port is None:
    try:
        print("Iniciando setup automático de comunicação Serial")
        serial_port_list = serial.tools.list_ports.comports()
        serial_port_list_size = len(serial_port_list)
        if (serial_port_list_size == 0):
            print ("Não foi detectado nenhuma comunicação serial compatível")
        elif (serial_port_list_size > 1):
            warnings.warn("ATENÇÃO - Foram encontradas mais de uma porta serial, o código exercutaa apenas com uma delas")
        selected_port = sorted(serial_port_list)[0]
        arduino = serial.Serial(port=selected_port.name, baudrate=115200, timeout=1)
        state = arduino.read()
        print(f"Porta {selected_port.name} conectada")
    except:
        print("Erro na conexão da comunicação serial, verifique se o encoder está conectado")
        exit()
else:
    try:
        arduino = serial.Serial(port=config.usb_com_port, baudrate=115200, timeout=1)
    except:
        print("Erro na conexão da comunicação serial, é recomendado alterar a variável usb_com_port no config.py para None")
        exit()


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


            print(f"Frame:  {frame_num:>3.2f}, delta:[{deltax:>5.2f},{deltay:>5.2f}], total_delta:[{total_deltax:>5.2f}, {total_deltay:>5.2f}]")


        frame_num = frame_num + 1
        img_processed_old = img_processed
    except KeyboardInterrupt:
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

