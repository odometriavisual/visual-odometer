import time
import warnings

import cv2
import serial.tools.list_ports

from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
from virtualencoder.visualodometry.svd_decomposition import optimized_svd_method
from virtualencoder.visualodometry.dsp_utils import cv2_to_nparray_grayscale, image_preprocessing

import config

# - Iniciar comunicação Serial - #
#
# if config.usb_com_port is None:
#     print("Iniciando setup automático de comunicação Serial")
#     serial_port_list = serial.tools.list_ports.comports()
#     serial_port_list_size = len(serial_port_list)
#     if (serial_port_list_size == 0):
#         print ("Não foi detectado nenhuma comunicação serial compatível")
#     elif (serial_port_list_size > 1):
#         warnings.warn("ATENÇÃO - Foram encontradas mais de uma porta serial, o código exercutaa apenas com uma delas")
#
#     selected_port = sorted(serial_port_list)[0]
#     arduino = serial.Serial(port=selected_port.name, baudrate=115200, timeout=.1)
#     print(f"Porta {selected_port.name} conectada")
# else:
#     arduino = serial.Serial(port=config.usb_com_port, baudrate=115200, timeout=.1)

# - Fim do setup Serial - #

print('Pegando acesso a camera, isso pode demorar um pouco...')
vid = cv2.VideoCapture(config.camera_id)
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

            #arduino.write(bytes(f"{multiplied_deltax, multiplied_deltay}", 'utf-8'))
            total_deltax = total_deltax + (multiplied_deltax)
            total_deltay = total_deltay + (multiplied_deltay)

            print(f"Frame:  {frame_num:>3.2f}, delta:[{deltax:>5.2f},{deltay:>5.2f}], total_delta:[{total_deltax:>5.2f}, {total_deltay:>5.2f}]")


        frame_num = frame_num + 1
        img_processed_old = img_processed
    except KeyboardInterrupt:
        passed_time = (time.time() - start_time)
        print("--- %s seconds ---" % passed_time)
        print("--- %s  frames ---" % frame_num)

        fps = frame_num / passed_time
        print("--- %s     fps ---" % fps)
        print("")
        print(f"Total deltax: {total_deltax}")
        print(f"Total deltay: {total_deltay}")

        vid.release()
