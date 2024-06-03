import datetime
import sys
import time
import warnings

import cv2
import numpy

import serial.tools.list_ports

import PIL.Image
import PIL.ImageOps
import flask

import concurrent.futures
import threading
import os
import json, os, signal

import webbrowser

#from flask import Flask, send_file, abort, render_template
import config

import serial

import scipy

from flask import Flask, render_template
import numpy as np

import plotly.graph_objs as go




#from mpl_toolkits.mplot3d import Axes3D

# Imports das bibliotecas necessarias para as bibliotecas
# imports dsp_utils.py

#from numpy.fft import fft2, fftshift


# imports dsp_utils.py
# imports image_utils.py



# imports image_utils.py
# imports svd_decomposition.py



# imports svd_decomposition.py



# imports dsp_utils.py
# imports image_utils.py

#codigo das bibliotecas

# imports dsp_utils.py
sys.path.extend([os.path.dirname(os.path.realpath(__file__))])
cont = True
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


def crosspower_spectrum(f, g, method=None):
    # Reference: https://en.wikipedia.org/wiki/Phase_correlation
    F = np.fft.fftshift(np.fft.fft2(f))
    G = np.fft.fftshift(np.fft.fft2(g))
    if method is not None:
        F = ideal_lowpass(F, method=method)
        G = ideal_lowpass(G, method=method)
    Q = F * np.conj(G) / np.abs(F * np.conj(G))
    q = np.real(np.fft.ifft2(Q))  # in theory, q must be fully real, but due to numerical approximations it is not.
    return q, Q


def ideal_lowpass2(I, method='Stone_et_al_2001'):
    if method == 'Stone_et_al_2001':
        m = 0.7 * I.shape[0]/2
        n = 0.7 * I.shape[1]/2
        N = np.min([m,n])
        I = I[int(I.shape[0] // 2 - N): int(I.shape[0] // 2 + N),
            int(I.shape[1] // 2 - N): int(I.shape[1] // 2 + N)]
        return I, N
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


def normalize_product(F: object, G: object, method="Stone_et_al_2001") -> object:
    # Versão modificada de crosspower_spectrum() para melhorias de eficiência

    Q = F * np.conj(G) / np.abs(F * np.conj(G))
    return Q




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



def generate_window(img_dim, window_type=None):
    if window_type is None:
        return np.ones(img_dim)
    elif window_type == 'Blackman-Harris':
        window1dy = scipy.signal.windows.blackmanharris(img_dim[0])
        window1dx = scipy.signal.windows.blackmanharris(img_dim[1])
        window2d = np.sqrt(np.outer(window1dy, window1dx))
        return window2d
    elif window_type == 'Blackman':
        window1dy = np.abs(np.blackman(img_dim[0]))
        window1dx = np.abs(np.blackman(img_dim[1]))
        window2d = np.sqrt(np.outer(window1dy, window1dx))
        return window2d


def apply_window(img1, img2, window_type):
    window = generate_window(img1.shape, window_type)
    return img1 * window, img2 * window


def autocontrast(img):
    autocontrast_img = img
    input_minval = autocontrast_img.min()
    input_maxval = autocontrast_img.max()
    output_img = (autocontrast_img - input_minval) / (input_maxval - input_minval) * 255
    return output_img


def gaussian_noise(image, snr):
    signal_energy = np.power(np.linalg.norm(image), 2)
    # snr = 20 * log10(signal_energy / noise_energy)
    # Therefore: noise_energy = 10^(- snr/20 + log10(signal_energy)
    noise_energy = np.power(10, np.log10(signal_energy) - snr / 20)

    # noise_energy ^2 =~ M * sigma^2 => sigma = sqrt(noise_energy^2 / M),
    # where sigma is the standard deviation and M is the signal length
    M = image.size  # Total number of elements
    sd = np.sqrt(np.power(noise_energy, 2) / M)
    sd = np.mean(image) * 0.2
    noise_signal = np.random.normal(0, sd, M)
    flattened_image = image.flatten(order="F") + noise_signal
    noisy_image = np.reshape(flattened_image, image.shape, order='F')
    return noisy_image


def salt_and_pepper(image, prob=0.05):
    # If the specified `prob` is negative or zero, we don't need to do anything.
    if prob <= 0:
        return image

    arr = np.asarray(image)
    original_dtype = arr.dtype

    # Derive the number of intensity levels from the array datatype.
    intensity_levels = 2 ** 8

    min_intensity = 0
    max_intensity = intensity_levels - 1

    # Generate an array with the same shape as the image's:
    # Each entry will have:
    # 1 with probability: 1 - prob
    # 0 or np.nan (50% each) with probability: prob
    random_image_arr = np.random.choice(
        [min_intensity, 1, np.nan], p=[prob / 2, 1 - prob, prob / 2], size=arr.shape
    )

    # This results in an image array with the following properties:
    # - With probability 1 - prob: the pixel KEEPS ITS VALUE (it was multiplied by 1)
    # - With probability prob/2: the pixel has value zero (it was multiplied by 0)
    # - With probability prob/2: the pixel has value np.nan (it was multiplied by np.nan)
    # We need to to `arr.astype(np.float)` to make sure np.nan is a valid value.
    salt_and_peppered_arr = arr.astype(np.float) * random_image_arr

    # Since we want SALT instead of NaN, we replace it.
    # We cast the array back to its original dtype so we can pass it to PIL.
    salt_and_peppered_arr = np.nan_to_num(
        salt_and_peppered_arr, nan=max_intensity
    ).astype(original_dtype)

    return salt_and_peppered_arr


def apply_blackman_harris_window(image):
    height, width = image.shape
    window_row = blackman_harris_window(width)
    window_col = blackman_harris_window(height)
    image_windowed = np.outer(window_col, window_row) * image
    return image_windowed


def blackman_harris_window(size, a0=0.35875, a1=0.48829, a2=0.14128, a3=0.01168):
    # a0, a1, a2 e a3 são os coeficientes de janelamento
    # Criação do vetor de amostras
    n = np.arange(size)
    # Cálculo da janela de Blackman-Harris
    window = a0 - a1 * np.cos(2 * np.pi * n / (size - 1)) + a2 * np.cos(4 * np.pi * n / (size - 1)) - a3 * np.cos(
        6 * np.pi * n / (size - 1))
    return window


def img_gen(dx=0, dy=0, width=512, height=600, zoom=1, angle=0):
    img = np.zeros((width, height), dtype=float)
    prng = numpy.random.RandomState(1234)
    circ = prng.rand(100, 4)

    # Definir a grade de coordenadas X e Y
    X = np.arange(width)
    Y = np.arange(height)

    # Aplicar o deslocamento
    X = X - (width) + dx
    Y = Y - (height) + dy

    # Aplicar o zoom
    X = (X / zoom) + (width / 2)
    Y = (Y / zoom) + (height / 2)

    X, Y = np.meshgrid(X, Y)

    for i in range(circ.shape[0]):
        # Calcular o ângulo de rotação para a máscara
        angle_rad = np.radians(angle)

        # Calcular as coordenadas X e Y ajustadas com rotação para cada ponto da malha
        X_rotated = X * np.cos(angle_rad) - Y * np.sin(angle_rad)
        Y_rotated = X * np.sin(angle_rad) + Y * np.cos(angle_rad)

        # Ajustar as coordenadas dos centros dos círculos para centralizar
        circle_center_x = circ[i, 0] * width - (width / 2)
        circle_center_y = circ[i, 1] * height - (height / 2)

        # Calcular a máscara com base nas coordenadas rotacionadas e ajustadas
        mask = ((X_rotated - circle_center_x) ** 2 + (Y_rotated - circle_center_y) ** 2) < (circ[i, 2] * 100) ** 2

        # Atribuir à imagem
        img[mask.T] = circ[i, 3]  # Transpor a máscara antes de atribuir à imagem

    return img


def add_gaussian_noise(img, noise_level=0.1):
    """
    Adiciona ruído gaussiano à imagem.

    Args:
        img (ndarray): Imagem de entrada.
        noise_level (float): Nível de ruído, padrão é 0.1.

    Returns:
        ndarray: Imagem com ruído gaussiano adicionado.
    """
    # Calcula a média e o desvio padrão da imagem
    mean = np.mean(img)
    std_dev = np.std(img)

    # Calcula o ruído gaussiano
    noise = np.random.normal(mean, std_dev * noise_level, img.shape)

    # Adiciona o ruído à imagem
    noisy_img = img + noise

    # Garante que os valores da imagem resultante estejam no intervalo [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255)

    return noisy_img.astype(np.uint8)  # Converte para o tipo de dados uint8


def add_blur(img, kernel_size=3):
    blurred_img = scipy.ndimage.uniform_filter(img.astype(float), size=kernel_size)
    return blurred_img  # Converte para o tipo de dados uint8


def add_noise(img, noise_type='gaussian', noise_level=0.1):
    if noise_type == 'gaussian':
        noisy_img = add_gaussian_noise(img, noise_level)
    else:
        raise ValueError("Tipo de ruído não suportado. Escolha entre 'gaussian', 'salt_and_pepper' ou 'speckle'.")
    return noisy_img


def filter_array_by_maxmin(arr):
    # Filtro para remover valores absolutos muito diferentes dentro de um array
    filtered_arr = []
    threshold = (arr.max() - arr.min()) / 4
    n = len(arr)

    # Se o array tiver menos que 3 elementos, não há o que filtrar
    if n < 3:
        return arr

    # Adiciona o primeiro elemento
    filtered_arr.append(arr[0])

    # Itera sobre os elementos do array, ignorando o primeiro e o último elemento
    for i in range(1, n - 1):
        diff_prev = abs(abs(arr[i]) - abs(arr[i - 1]))
        diff_next = abs(abs(arr[i]) - abs(arr[i + 1]))

        # Verifica se a diferença entre o elemento atual e o anterior, e entre o atual e o próximo
        # são menores que o threshol
        if (diff_prev < threshold and diff_next < threshold):
            filtered_arr.append(arr[i])
        else:
            median = arr[i - 1] / 2 + arr[i + 1] / 2
            if abs(median - arr[i - 1]) < threshold:
                filtered_arr.append(median)

    # Adiciona o último elemento
    filtered_arr.append(arr[-1])
    return np.array(filtered_arr)


def apply_raised_cosine_window(image):
    rows, cols = image.shape
    i = np.arange(rows)
    j = np.arange(cols)
    window = 0.5 * (1 + np.cos(np.pi * (2 * i[:, None] - rows) / rows)) * \
             0.5 * (1 + np.cos(np.pi * (2 * j - cols) / cols))
    return image * window


def apply_border_windowing_on_image(image, border_windowing_method="blackman_harris"):
    if border_windowing_method == "blackman_harris":
        return apply_blackman_harris_window(image)
    elif border_windowing_method == "raised_cosine":
        return apply_raised_cosine_window(image)
    elif border_windowing_method == None:
        return image


# imports image_utils.py
# imports svd_decomposition.py

def linear_regression(x, y):
    R = np.ones((x.size, 2))
    R[:, 0] = x
    mu, c = np.linalg.inv((R.transpose() @ R)) @ R.transpose() @ y
    return mu, c


def phase_unwrapping(phase_vec, factor=0.7):
    phase_diff = np.diff(phase_vec)
    corrected_difference = phase_diff - 2. * np.pi * (phase_diff > (2 * np.pi * factor)) + 2. * np.pi * (
                phase_diff < -(2 * np.pi * factor))
    return np.cumsum(corrected_difference)


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


def svd_method(img_beg, img_end, frequency_window="Stone_et_al_2001"):
    M, N = img_beg.shape
    q, Q = crosspower_spectrum(img_end, img_beg, frequency_window)
    qu, s, qv = scipy.sparse.linalg.svds(Q, k=1)
    ang_qu = np.angle(qu[:, 0])
    ang_qv = np.angle(qv[0, :])

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay = svd_estimate_shift(ang_qu, M)
    deltax = svd_estimate_shift(ang_qv, N)
    return deltax, deltay

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

# imports svd_decomposition.py
# imports score_focus.py


def score_lapv(img):
    """Implements the Variance of Laplacian (LAP4) focus measure
    operator. Measures the amount of edges present in the image.

    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    return numpy.std(cv2.Laplacian(img, cv2.CV_64F)) ** 2


def score_lapm(img):
    """Implements the Modified Laplacian (LAP2) focus measure
    operator. Measures the amount of edges present in the image.

    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    kernel = numpy.array([-1, 2, -1])
    laplacianX = numpy.abs(cv2.filter2D(img, -1, kernel))
    laplacianY = numpy.abs(cv2.filter2D(img, -1, kernel.T))
    return numpy.mean(laplacianX + laplacianY)


def score_teng(img):
    """Implements the Tenengrad (TENG) focus measure operator.
    Based on the gradient of the image.

    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    return numpy.mean(gaussianX * gaussianX +
                      gaussianY * gaussianY)


def score_mlog(img):
    """Implements the MLOG focus measure algorithm.

    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    return numpy.max(cv2.convertScaleAbs(cv2.Laplacian(img, 3)))

# imports score_focus.py


# Configurações de câmera
camera_id = 0           # Defina o id da câmera
camera_exposure = -11   # Defina exposição da câmera

# Multiplicadores de deslocamento
deltax_multiplier = 1 # Defina o multiplicador de deslocamento X
deltay_multiplier = 1 # Defina o multiplicador de deslocamento Y

# Configuração de comunicação Serial
usb_com_port = None  # Configure a porta de comunicação serial (Padrão: None, Valores Possíveis: String, Exemplo: "COM4")

# Configurações de estimativa
border_windowing_method = "blackman_harris"  # Aplica o escurecimento nas bordas das imagens
phase_windowing = None                       # Aplica o janelamento no sinal final da fase
# ----- Configuração de comunicação Serial ----- #


# Obtém o diretório atual do arquivo Python
dir_path = os.path.dirname(os.path.realpath(__file__))

# Caminho para o arquivo HTML
html_file_path = os.path.join(dir_path, 'executar.html')



app = flask.Flask(__name__)
app.template_folder = ''

lock_quat = threading.Lock()
printar = False
lista_dados = []  # este é o Y
lista_dados2 = []  # este é o X
lista_imu = []
lista_3d = []
all_points_3d = []
all_points_2d = []

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_atual = os.getcwd()


global data_dir
data_dir = os.path.join(diretorio_atual, 'data')

# Verifica se o diretório "data" existe
if os.path.isdir(data_dir):
     files = os.listdir(data_dir)
    # for file_name in files:
    #     file_path = os.path.join(data_dir, file_name)
    #     os.remove(file_path)
    # os.rmdir(data_dir)
    # os.mkdir(data_dir)
else:
    os.mkdir(data_dir)





#from virtualencoder.visualodometry.score_focus import score_teng

vid = cv2.VideoCapture(camera_id)
total_rec_time = 60  # seconds
max_fps = 30  # Define o FPS máximo desejado

score_history = [0] * 270
counter = 0 #valor padrão do foco

vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
vid.set(cv2.CAP_PROP_FOCUS, 255)

serial_pulsador = 0

gyroData = [0,0,0,0]
glob_quat = [0, 1, 0, 0]
counter = 0
offset = None

# --- Funções reservadas para envio e armazenamento de dados --- #

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def checkSerialInput(first_time = False):
    global gyroData, serial_giroscopio, offset
    if (serial_giroscopio.in_waiting > 0):
        ser_line = serial_giroscopio.readline().decode()
        gyroData = [float(x) for x in ser_line.split(",")] #list  quaternium to biallhole
        gyroData = [gyroData[0], gyroData[1], gyroData[2], gyroData[3]]
        serial_giroscopio.read_all()

        #if first_time is True:
            #quat_first = [quat[0], -quat[1], -quat[2], -quat[3]]
            #offset = quaternion_multiply(glob_quat, quat_first)

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

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])


def minha_thread():
    global printar
    global vid
    global resto_x
    global resto_y
    global lista_imu
    global lista_3d
    global intxacumulado
    global intyacumulado
    global serial_pulsador
    global serial_giroscopio
    global ddd
    global cont
    global all_points_3d
    global all_points_2d
    global final_3d

    totalx = 0
    totaly = 0
    giroscop = 0
    # --- definição de variáveis necessárias --- #

    total_deltax = 0
    total_deltay = 0

    M = None
    N = None
    start_time = 0

    resto_x = 0
    resto_y = 0

    intxacumulado = 0
    intyacumulado = 0

    ports = serial.tools.list_ports.comports()
    print("ports = ", ports)
    for port in ports:
        print(port)
        if port.serial_number == "56CA000930" or port.serial_number == "5598007147" or port.serial_number == "5698011281" or port.serial_number == "5767002927":
            print("Iniciando conecção com o modulo do giroscópio")
            serial_giroscopio = serial.Serial(port=port.device, baudrate=115200, timeout=1)
            serial_giroscopio.setRTS(False)
            time.sleep(0.3)
            serial_giroscopio.setRTS(True)
            time.sleep(0.3)
            serial_giroscopio.setRTS(False)
            time.sleep(0.3)
        elif port.serial_number == "5767003473":
            print("Iniciando comunicação com o modulo pulsador")
            serial_pulsador = serial.Serial(port=port.device, baudrate=115200, timeout=1)
            serial_pulsador.setRTS(False)
            time.sleep(0.3)
            serial_pulsador.setRTS(True)
            time.sleep(0.3)
            serial_pulsador.setRTS(False)
            time.sleep(0.3)

    print("Testando comunicação serial: encoder")
    _ = serial_pulsador.read()
    print("Testando comunicação serial: giroscópio server")
    _ = serial_giroscopio.read()
    print("Comunicação serial OK")
    print("")

    # -- Inicio da configuração da câmera --- #

    print('Pegando acesso a camera, isso pode demorar um pouco...')
    ##vid = cv2.VideoCapture(camera_id)


    if camera_exposure != None:
        print("Definindo exposição da câmera")
        vid.set(cv2.CAP_PROP_EXPOSURE, camera_exposure)

    frame_num = -10
    # Abre o arquivo HTML no navegador padrão

    webbrowser.open(html_file_path)


    print("cont = ", cont)
    i = 0






    first_time = True
    point_3d = np.zeros(shape=3)
    point_2d = np.zeros(shape=3)
    print("antes do loop")
    while cont:
        try:
            # Comentado para remover serial
            # print("checkSerialInput()")
            checkSerialInput(first_time)
            first_time = False

            ret, frame = vid.read()
            img_array = cv2_to_nparray_grayscale(frame)
            img_windowed = apply_border_windowing_on_image(img_array, border_windowing_method)
            img_processed = image_preprocessing(img_array)
            if frame_num > 0:
                if (M == None):
                    print("Script iniciado")
                    start_time = time.time()
                    M, N = img_array.shape
                deltax, deltay = optimized_svd_method(img_processed, img_processed_old, M, N,
                                                      phase_windowing=phase_windowing)
                multiplied_deltax = deltax_multiplier * deltax
                multiplied_deltay = deltay_multiplier * deltay


                # Comentado para remover serial
                serialSendEncoder(multiplied_deltax,
                                  multiplied_deltay)  # <- Envia informações de deslocamento para o modulo pulsador

                total_deltax = total_deltax + multiplied_deltax
                total_deltay = total_deltay + multiplied_deltay





                # Exemplo de array para ser salvo:
                array_to_save = [time.time(), gyroData, total_deltax, total_deltay]
                print(array_to_save)

                # salvando x e y
                # para o web
                # Recomendo limitar o salvamento a um intervalo. Mas é só uma sugestão mesmo.




                totaly = round(total_deltay, 2)
                totalx = round(total_deltax, 2)
                ##print("test")

                rotation_matrix = quaternion_to_rotation_matrix(gyroData)
                # angles = quaternion_to_euler(gyroData)
                # print(angles)
                point_3d = point_3d + np.dot(rotation_matrix, [multiplied_deltay, multiplied_deltax, 0 ])
                point_2d = point_2d + [multiplied_deltax, multiplied_deltay, 0]
                ##print("test1")

                # r = scipy.spatial.transform.Rotation.from_quat([gyroData])
                # v = [totalx, totaly, 0]
                # ddd = r.apply(v)
                ## alterar a forma que o ddd esta gerando os dados para a analise de deslocamento ser mais rapida


                lock_quat.acquire()
                if printar:
                    lista_dados2.append(totalx)
                    lista_dados.append(totaly)
                    lista_imu.append(gyroData)
                    all_points_3d.append(point_3d)
                    all_points_2d.append(point_2d)
                    ##print(point_3d)


                lock_quat.release()



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
        except Exception as exc:
            print("Erro:", exc)
        i += 1
        ## print('i = ', i)
        # if i > 50:
        #     break
    print("aqui")
    return ""

def quaternion_to_euler(quat):
    w, x, y, z = quat
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.degrees(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.degrees(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.degrees(np.arctan2(t3, t4))

    return X, Y, Z

# def salvar_dados_arquivo():
#     global all_points_3d, lista_dados2, lista_imu, lista_3d
#     print(lista_3d)
#     print("lista_3d:")
#     dados_lista = [arr.tolist() for arr in lista_3d]
#     print(dados_lista)
#
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Cria um timestamp único
#     arquivo_dados = os.path.join(data_dir, f"dados_{timestamp}.txt")  # Nome do arquivo com timestamp
#     print(arquivo_dados)
#     with open(arquivo_dados, "w") as arquivo:
#         for x, y, z in zip(lista_dados, lista_dados2, lista_3d):
#             arquivo.write(
#                 f"{x}|{y}|{z}\n")  # Escreve os dados x e y em uma linha, separados por um espaço e com uma quebra de linha no final


def salvar_dados_arquivo():
    global all_points_3d, all_points_2d



    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Cria um timestamp único
    arquivo_dados = os.path.join(data_dir, f"dados_{timestamp}.txt")  # Nome do arquivo com timestamp

    with open(arquivo_dados, "w") as arquivo:
        for points_2d, points_3d in zip(all_points_2d, all_points_3d):
            # Convertendo os pontos 3D em uma string separada por vírgula
            points_3d_str = ','.join(map(str, points_3d))
            # Escrevendo os pontos 2D e 3D no arquivo, separados por |
            arquivo.write(f"{','.join(map(str, points_2d))}|{points_3d_str}\n")

@app.route('/iniciar', methods=["GET", "POST"])
def iniciar():
    global printar

    printar = True
    return '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;"><a href="/3D"><button style="width: 150px; height: 50px;background;">3D</button></a><form action="/finalizar" method="post"><button type="submit" style="width: 150px; height: 50px;">stop</button></form></div>'

@app.route('/3D')
def vizu3d():
    global all_points_3d
    global all_points_2d

    final_3d = [list(arr) for arr in all_points_3d]
    final_2d = [list(arr) for arr in all_points_2d]


    # Criar um gráfico de dispersão 3D
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        )
    )


    # Criar a figura do gráfico
    fig = go.Figure(layout=layout)
    fig.add_trace(
        go.Scatter3d(
            x=[point[0] for point in final_3d],
            y=[point[1] for point in final_3d],
            z=[point[2] for point in final_3d],
            mode='markers',

            marker=dict(
                size=12,
                color='blue',
                opacity=0.8,

            )
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[point[0] for point in final_2d],
            y=[point[1] for point in final_2d],
            z=[point[2] for point in final_2d],
            mode='markers',

            marker=dict(
                size=12,
                color='red',
                opacity=0.8,
            )
        )
    )

    fig.layout

    # Converter a figura para HTML
    graph_html = fig.to_html(full_html=False)

    # Renderizar o template HTML com o gráfico
    return graph_html


@app.route('/finalizar', methods=["GET", "POST"])
def finalizar():
    print("teste")
    global printar
    global lista_dados, lista_dados2, lista_imu, lista_3d
    salvar_dados_arquivo()
    lista_dados = []  # Limpa a lista após salvar os dados
    lista_dados2 = []
    lista_imu = []
    lista_3d = []
    printar = False
    return '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;"><form action="/iniciar" method="post"><button type="submit" style="width: 150px; height: 50px;">start</button></form><br><form action="/dados" method="post"><button type="submit" style="width: 150px; height: 50px;">results</button></form></div>'

# def alterarVariavel(valor):
#
#     minhaVariavel = valor
#
#     return minhaVariavel

@app.route('/excluir/<nome_arquivo>', methods=["GET", "POST"])
def excluirArquivo(nome_arquivo):
    global data_dir
    caminho_arquivo = os.path.join(data_dir, nome_arquivo)

    # Verifica se o arquivo existe
    if os.path.exists(caminho_arquivo):
        # Remove o arquivo
        os.remove(caminho_arquivo)
    return flask.redirect(flask.url_for('mostrar_dados'))
@app.route('/3D/<nome_arquivo>', methods=["GET", "POST"])
def abrirArquivo(nome_arquivo):
    dd = []
    ddd = []
    try:
        arquivo_path = os.path.join(data_dir, nome_arquivo)
        if os.path.isfile(arquivo_path):
            with open(arquivo_path, "r") as arquivo:
                for linha in arquivo:
                    numeros = linha.strip().split('|')
                    dois = numeros[0].strip().split(',')
                    tres = numeros[1].strip().split(',')
                    float_dois = [float(x) for x in dois]
                    float_tres = [float(x) for x in tres]
                    dd.append(float_dois)
                    ddd.append(float_tres)
                # fim abertura arquivo
                # Criar um gráfico de dispersão 3D
                layout = go.Layout(
                    scene=dict(
                        xaxis=dict(title='X'),
                        yaxis=dict(title='Y'),
                        zaxis=dict(title='Z'),
                          aspectmode='data'
                    )
                   )

                # Criar a figura do gráfico
                fig = go.Figure(layout=layout)
                fig.add_trace(
                    go.Scatter3d(
                        x=[point[0] for point in ddd],
                        y=[point[1] for point in ddd],
                        z=[point[2] for point in ddd],
                        mode='markers',

                        marker=dict(
                            size=12,
                            color='blue',
                            opacity=0.8,

                        )
                    )
                )

                # fig.add_trace(
                #     go.Scatter3d(
                #         x=[point[0] for point in dd],
                #         y=[point[1] for point in dd],
                #         z=[point[2] for point in dd],
                #         mode='markers',
                #
                #         marker=dict(
                #             size=12,
                #             color='red',
                #             opacity=0.8,
                #         )
                #     )
                # )

                fig.layout

                # Converter a figura para HTML
                graph_html = fig.to_html(full_html=False)
                # Renderizar o template HTML com o gráfico
                return graph_html


        else:
            flask.abort(
                404)  # Retorna um erro 404 se o arquivo não existir  <form action="/download" method="post"><button type="submit" style="width: 150px; height: 50px;">download</button>

    except Exception as e:

        return "Erro interno do servidor", 500  # Retorna um erro 500 se ocorrer uma exceção

@app.route('/dados', methods=["GET", "POST"])
def mostrar_dados():
    html = '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;">'

    html += '<br><br><form action="/iniciar" method="post"><button type="submit" style="width: 150px; height: 50px;">start</button></form><br><form action="/parar" method="post"><button type="submit" style="width: 150px; height: 50px;">finish prog</button></form>'

    # Listar todos os arquivos .txt no diretório atual
    arquivos_txt = [arquivo for arquivo in os.listdir(data_dir) if arquivo.endswith(".txt")]

    # Criar botões para cada arquivo .txt
    for arquivo in arquivos_txt:
        html += f'<div style="vertical-align: middle;"><tr><td>{arquivo}</td>	<td style="vertical-align: middle;"><a href="/3D/{arquivo}"><button style="width: 90px; height: 20px;background;">Visualizar</button></a></td>	<td style="vertical-align: middle;"><a href="/abrir_arquivo/{arquivo}"><button style="width: 90px; height: 20px;background;">Baixar</button></a></td>	<td style="vertical-align: middle;"><a href="/excluir/{arquivo}"><button style="width: 90px; height: 20px;background;">Excluir</button><br></a></td></tr></div>'
    html += '</div>'
    return html

@app.route('/abrir_arquivo/<nome_arquivo>', methods=["GET", "POST"])
def abrir_arquivo(nome_arquivo):
    try:
        arquivo_path = os.path.join(data_dir, nome_arquivo)
        if os.path.isfile(arquivo_path):
            with open(arquivo_path, "r") as arquivo:
                conteudo = arquivo.read()
            return f'<div style="white-space: pre-line;">{conteudo}<script>window.location.href = "http://localhost:5000/download/{nome_arquivo}"</script></div>'
        else:
            flask.abort(
                404)  # Retorna um erro 404 se o arquivo não existir  <form action="/download" method="post"><button type="submit" style="width: 150px; height: 50px;">download</button>

    except Exception as e:

        return "Erro interno do servidor", 500  # Retorna um erro 500 se ocorrer uma exceção

@app.route('/download/<nome_arquivo>', methods=["GET"])
def download(nome_arquivo):
    try:
        arquivo_path = os.path.join(data_dir, nome_arquivo)
        if os.path.isfile(arquivo_path):
            return flask.send_file(arquivo_path, as_attachment=True)
        else:
            flask.abort(404)  # Retorna um erro 404 se o arquivo não existir
    except Exception as e:
        return "Erro interno do servidor", 500  # Retorna um erro 500 se ocorrer uma exceção

@app.route('/autofoco', methods=["GET", "POST"])
def autofoco():
    global vid
    global counter
    global score_history


    while counter < 260:
        ret, frame = vid.read()
        cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        focus_score = score_teng(cv2_img)
        score_history[counter] = focus_score
        counter += 5
        vid.set(cv2.CAP_PROP_FOCUS, counter)

    vid.set(cv2.CAP_PROP_FOCUS, np.argmax(score_history))
    return f'<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;">{np.argmax(score_history)}<form action="/iniciar" method="post"><button type="submit" style="width: 150px; height: 50px;">start</button></form><br><form action="/dados" method="post"><button type="submit" style="width: 150px; height: 50px;">results</button></form></div>'

@app.route('/parar', methods=["GET", "POST"])
def parar():
    global cont

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

    cont = False
    os.kill(os.getpid(), signal.SIGINT)

    return "Finished"

if __name__ == "__main__":
    thread = threading.Thread(target=minha_thread)
    thread.start()
    app.run(host="127.0.0.1", port=5000)
    thread.join()

