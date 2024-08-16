import sys
import time
import warnings

import cv2

from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
from virtualencoder.visualodometry.svd_decomposition import optimized_svd_method
from virtualencoder.visualodometry.dsp_utils import cv2_to_nparray_grayscale, image_preprocessing

import config

vid = cv2.VideoCapture('http://10.42.0.95:7123/stream.mjpg')
try:
    ret, frame = vid.read()
    img_array = cv2_to_nparray_grayscale(frame)
except:
    print("Não foi possível conectar a câmera, altere o id da camera no config.py")
    exit()

frame_num = -10 #Definido para negativo, o frame será contabilizado apenas após a décima imagem
M = None
N = None

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
            total_deltax = total_deltax + (config.deltax_multiplier * deltax)
            total_deltay = total_deltay + (config.deltay_multiplier * deltay)
            print(f"Frame:  {frame_num:>3.2f}, delta:[{deltax:>5.2f},{deltay:>5.2f}], total_delta:[{total_deltax:>5.2f}, {total_deltay:>5.2f}]")
        frame_num = frame_num + 1
        img_processed_old = img_processed
    except:
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

