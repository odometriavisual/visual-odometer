import sys
import time
import warnings
import numpy as np

#sys.path.insert(0, r'C:\Users\Panther\PycharmProjects\virtualencoder')

#w = "..\pasta"
import cv2
import serial.tools.list_ports

#from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
#from virtualencoder.visualodometry.svd_decomposition import optimized_svd_method
#from virtualencoder.visualodometry.dsp_utils import cv2_to_nparray_grayscale, image_preprocessing

#import scipy.ndimage.convolve
#import scipy.sparse.linalg.svds

#import scipy
import PIL.Image
import PIL.ImageOps

camera_id = 0           # Defina o id da câmera
camera_exposure = None   # Defina exposição da câmera

# Multiplicadores de deslocamento
deltax_multiplier = 1 # Defina o multiplicador de deslocamento X
deltay_multiplier = 1 # Defina o multiplicador de deslocamento Y

# Configuração de comunicação Serial
usb_com_port = None  # Configure a porta de comunicação serial (Padrão: None, Valores Possíveis: String, Exemplo: "COM4")

# Configurações de estimativa
border_windowing_method = "blackman_harris"  # Aplica o escurecimento nas bordas das imagens
phase_windowing = None                       # Aplica o janelamento no sinal final da fase

#import scipy.ndimage
#import scipy.sparse.linalg

import scipy    

#from scipy.ndimage import convolve
#from scipy.sparse.linalg import svds
#from PIL import Image, ImageOps
#from numpy.fft import fft2, fftshift

def ideal_lowpass(I, factor=0.6, method='Stone_et_al_2001'):
    if method == 'Stone_et_al_2001':
        m = factor * I.shape[0]/2
        n = factor * I.shape[1]/2
        N = np.min([m, n])
        I = I[int(I.shape[0] // 2 - N): int(I.shape[0] // 2 + N),
            int(I.shape[1] // 2 - N): int(I.shape[1] // 2 + N)]
        return I
    else:
        raise ValueError('Método não suportado.')

def image_preprocessing(image, method='Stone_et_al_2001'):
    fft_from_image = np.fft.fftshift(np.fft.fft2(image))
    if method is not None:
        fft_from_image = ideal_lowpass(fft_from_image, method=method)
    return fft_from_image

def cv2_to_nparray_grayscale(frame):
    cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_array_rgb = PIL.Image.fromarray(cv2_img)
    img_grayscale = PIL.ImageOps.grayscale(img_array_rgb)
    img_array = np.asarray(img_grayscale)
    return img_array
def phase_unwrapping(phase_vec, factor=0.7):
    phase_diff = np.diff(phase_vec)
    corrected_difference = phase_diff - 2. * np.pi * (phase_diff > (2 * np.pi * factor)) + 2. * np.pi * (
                phase_diff < -(2 * np.pi * factor))
    return np.cumsum(corrected_difference)

def linear_regression(x, y):
    R = np.ones((x.size, 2))
    R[:, 0] = x
    mu, c = np.linalg.inv((R.transpose() @ R)) @ R.transpose() @ y
    return mu, c

def svd_estimate_shift(phase_vec, N, phase_windowing=None):
    # Phase unwrapping:
    phase_unwrapped = phase_unwrapping(phase_vec)
    r = np.arange(0, phase_unwrapped.size)
    M = r.size // 2
    if phase_windowing == None or phase_windowing == False:
        x = r
        y = phase_unwrapped
    elif phase_windowing == "central":
        x = r[M - 50:M + 50]
        y = phase_unwrapped[M - 50:M + 50]
    elif phase_windowing == "initial":
        x = r[M - 80:M - 10]
        y = phase_unwrapped[M - 80:M - 10]
    mu, c = linear_regression(x, y)
    delta = mu * N / (2 * np.pi)
    return delta

def phase_fringe_filter(cross_power_spectrum, window_size=(5, 5), threshold=0.03):
    # Aplica o filtro de média para reduzir o ruído
    filtered_spectrum = scipy.ndimage.convolve(cross_power_spectrum, np.ones(window_size) / np.prod(window_size), mode='constant')

    # Calcula a diferença entre o espectro original e o filtrado
    diff_spectrum = cross_power_spectrum - filtered_spectrum

    # Aplica o limiar para identificar as regiões de pico
    peak_mask = np.abs(diff_spectrum) > threshold

    # Atenua as regiões de pico no espectro original
    phase_filtered_spectrum = cross_power_spectrum.copy()
    phase_filtered_spectrum[peak_mask] *= 0.5  # Reduz a amplitude nas regiões de pico

    return phase_filtered_spectrum
def normalize_product(F: object, G: object, method="Stone_et_al_2001") -> object:
    # Versão modificada de crosspower_spectrum() para melhorias de eficiência

    Q = F * np.conj(G) / np.abs(F * np.conj(G))
    return Q

def optimized_svd_method(processed_img_beg, processed_img_end, M, N, phase_windowing=None, finge_filter = True):
    Q = normalize_product(processed_img_end, processed_img_beg)
    if finge_filter is True:
        Q = phase_fringe_filter(Q)

    qu, s, qv = scipy.sparse.linalg.svds(Q, k=1)
    ang_qu = np.angle(qu[:, 0])
    ang_qv = np.angle(qv[0, :])

    # if filter_values: # desativado, resolução inicial do bug nomeado de f153
    #     ang_qu = filter_array_by_maxmin(ang_qu)
    #     ang_qv = filter_array_by_maxmin(ang_qv)

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay = svd_estimate_shift(ang_qu, M, phase_windowing)
    deltax = svd_estimate_shift(ang_qv, N, phase_windowing)
    return deltax, deltay
def apply_raised_cosine_window(image):
    rows, cols = image.shape
    i = np.arange(rows)
    j = np.arange(cols)
    window = 0.5 * (1 + np.cos(np.pi * (2 * i[:, None] - rows) / rows)) * \
             0.5 * (1 + np.cos(np.pi * (2 * j - cols) / cols))
    return image * window
def blackman_harris_window(size, a0=0.35875, a1=0.48829, a2=0.14128, a3=0.01168):
    # a0, a1, a2 e a3 são os coeficientes de janelamento
    # Criação do vetor de amostras
    n = np.arange(size)
    # Cálculo da janela de Blackman-Harris
    window = a0 - a1 * np.cos(2 * np.pi * n / (size - 1)) + a2 * np.cos(4 * np.pi * n / (size - 1)) - a3 * np.cos(
        6 * np.pi * n / (size - 1))
    return window

def apply_blackman_harris_window(image):
    height, width = image.shape
    window_row = blackman_harris_window(width)
    window_col = blackman_harris_window(height)
    image_windowed = np.outer(window_col, window_row) * image
    return image_windowed
def apply_border_windowing_on_image(image, border_windowing_method="blackman_harris"):
    if border_windowing_method == "blackman_harris":
        return apply_blackman_harris_window(image)
    elif border_windowing_method == "raised_cosine":
        return apply_raised_cosine_window(image)
    elif border_windowing_method == None:
        return image



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
vid = cv2.VideoCapture(camera_id)
ret, frame = vid.read()
img_array = cv2_to_nparray_grayscale(frame)

if camera_exposure != None:
    print("Definindo exposição da câmera")
    vid.set(cv2.CAP_PROP_EXPOSURE, camera_exposure)

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
        img_windowed = apply_border_windowing_on_image(img_array, border_windowing_method)
        img_processed = image_preprocessing(img_array)
        if frame_num > 0:
            if (M == None):
                print("Script iniciado")
                start_time = time.time()
                M, N = img_array.shape
            deltax, deltay = optimized_svd_method(img_processed, img_processed_old, M, N, phase_windowing=phase_windowing, finge_filter=False)
            multiplied_deltax = deltax_multiplier * deltax
            multiplied_deltay = deltay_multiplier * deltay
            atualizarPos(multiplied_deltax,multiplied_deltay)
            total_deltax = total_deltax + (multiplied_deltax)
            total_deltay = total_deltay + (multiplied_deltay)

            print(f"Frame:  {frame_num:>3.2f}, delta:[{deltax:>5.2f},{deltay:>5.2f}], total_delta:[{total_deltax:>5.2f}, {total_deltay:>5.2f}, {gyroData}]")

        frame_num = frame_num + 1
        img_processed_old = img_processed
    except Exception as e:
        print(e)
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

