
from flask import Flask, send_file, abort, render_template
import concurrent.futures
import threading
import os
import time
import datetime
import webbrowser

import sys
import warnings

import cv2
import serial.tools.list_ports

import config

import matplotlib.pyplot as plt
import numpy as np

import serial
import serial.tools.list_ports

import time

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Imports das bibliotecas necessarias para as bibliotecas
# imports dsp_utils.py

from numpy.fft import fft2, fftshift
import cv2
from PIL import Image, ImageOps

# imports dsp_utils.py
# imports image_utils.py

import numpy as np
from numpy.random import RandomState
from scipy import signal
from scipy.ndimage import uniform_filter

# imports image_utils.py
# imports svd_decomposition.py

from scipy.sparse.linalg import svds

# imports svd_decomposition.py

#codigo das bibliotecas

# imports dsp_utils.py

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
    F = fftshift(fft2(f))
    G = fftshift(fft2(g))
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
    fft_from_image = fftshift(fft2(image))
    if method is not None:
        fft_from_image = ideal_lowpass(fft_from_image, method=method)
    return fft_from_image

def cv2_to_nparray_grayscale(frame):
    cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_array_rgb = Image.fromarray(cv2_img)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    img_array = np.asarray(img_grayscale)
    return img_array


def normalize_product(F: object, G: object, method="Stone_et_al_2001") -> object:
    # Versão modificada de crosspower_spectrum() para melhorias de eficiência

    Q = F * np.conj(G) / np.abs(F * np.conj(G))
    return Q

import numpy as np
from scipy.ndimage import convolve



def phase_fringe_filter(cross_power_spectrum, window_size=(5, 5), threshold=0.03):
    # Aplica o filtro de média para reduzir o ruído
    filtered_spectrum = convolve(cross_power_spectrum, np.ones(window_size) / np.prod(window_size), mode='constant')

    # Calcula a diferença entre o espectro original e o filtrado
    diff_spectrum = cross_power_spectrum - filtered_spectrum

    # Aplica o limiar para identificar as regiões de pico
    peak_mask = np.abs(diff_spectrum) > threshold

    # Atenua as regiões de pico no espectro original
    phase_filtered_spectrum = cross_power_spectrum.copy()
    phase_filtered_spectrum[peak_mask] *= 0.5  # Reduz a amplitude nas regiões de pico

    return phase_filtered_spectrum

# imports dsp_utils.py
# imports image_utils.py

def generate_window(img_dim, window_type=None):
    if window_type is None:
        return np.ones(img_dim)
    elif window_type == 'Blackman-Harris':
        window1dy = signal.windows.blackmanharris(img_dim[0])
        window1dx = signal.windows.blackmanharris(img_dim[1])
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
    prng = RandomState(1234)
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
    blurred_img = uniform_filter(img.astype(float), size=kernel_size)
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
    qu, s, qv = svds(Q, k=1)
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

    qu, s, qv = svds(Q, k=1)
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






# Configurações de câmera
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
# ----- Configuração de comunicação Serial ----- #


# Obtém o diretório atual do arquivo Python
dir_path = os.path.dirname(os.path.realpath(__file__))

# Caminho para o arquivo HTML
html_file_path = os.path.join(dir_path, 'executar.html')



app = Flask(__name__)
app.template_folder = ''

lock_quat = threading.Lock()
printar = False
lista_dados = []  # este é o Y
lista_dados2 = []  # este é o X
lista_imu = []
lista_3d = []






diretorio_atual = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(diretorio_atual, "data")
files = os.listdir(data_dir)
for file_name in files:
    file_path = os.path.join(data_dir, file_name)
    os.remove(file_path)

from virtualencoder.visualodometry.score_focus import score_teng

vid = cv2.VideoCapture(camera_id)
camera_id = 0  # Altere o id da câmera aqui
total_rec_time = 60  # seconds
max_fps = 30  # Define o FPS máximo desejado
camera_exposure = -6  # Defina exposição da câmera

score_history = [0] * 270
counter = 0

vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
vid.set(cv2.CAP_PROP_FOCUS, counter)

serial_pulsador = 0

gyroData = [0,0,0,0]

# --- Funções reservadas para envio e armazenamento de dados --- #

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

def checkSerialInput():
    global gyroData, serial_giroscopio
    if (serial_giroscopio.in_waiting > 0):
        ser_line = serial_giroscopio.readline().decode()
        gyroData = [float(x) for x in ser_line.split(",")] #list  quaternium to biallhole
        serial_giroscopio.flush()

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


def minha_thread():
    from flask import Flask, send_file, abort, render_template
    import concurrent.futures
    import threading
    import os
    import time
    import datetime
    import webbrowser

    import sys
    import warnings

    sys.path.insert(0, r'C:\Users\Panther\PycharmProjects\virtualencoder')

    import cv2
    import serial.tools.list_ports

    from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
    from virtualencoder.visualodometry.svd_decomposition import optimized_svd_method
    from virtualencoder.visualodometry.dsp_utils import cv2_to_nparray_grayscale, image_preprocessing

    import config

    import matplotlib.pyplot as plt
    import numpy as np

    import serial
    import serial.tools.list_ports

    import time

    from scipy.spatial.transform import Rotation as R

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # Configurações de câmera
    camera_id = 0  # Defina o id da câmera
    camera_exposure = None  # Defina exposição da câmera

    # Multiplicadores de deslocamento
    deltax_multiplier = 1  # Defina o multiplicador de deslocamento X
    deltay_multiplier = 1  # Defina o multiplicador de deslocamento Y

    # Configuração de comunicação Serial
    usb_com_port = None  # Configure a porta de comunicação serial (Padrão: None, Valores Possíveis: String, Exemplo: "COM4")

    # Configurações de estimativa
    border_windowing_method = "blackman_harris"  # Aplica o escurecimento nas bordas das imagens
    phase_windowing = None  # Aplica o janelamento no sinal final da fase
    # ----- Configuração de comunicação Serial ----- #

    # Obtém o diretório atual do arquivo Python
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Caminho para o arquivo HTML
    html_file_path = os.path.join(dir_path, 'executar.html')

    app = Flask(__name__)
    app.template_folder = ''

    lock_quat = threading.Lock()
    printar = False
    lista_dados = []  # este é o Y
    lista_dados2 = []  # este é o X
    lista_imu = []
    lista_3d = []

    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(diretorio_atual, "data")
    files = os.listdir(data_dir)
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        os.remove(file_path)

    from virtualencoder.visualodometry.score_focus import score_teng

    vid = cv2.VideoCapture(camera_id)
    camera_id = 0  # Altere o id da câmera aqui
    total_rec_time = 60  # seconds
    max_fps = 30  # Define o FPS máximo desejado
    camera_exposure = -6  # Defina exposição da câmera

    score_history = [0] * 270
    counter = 0

    vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    vid.set(cv2.CAP_PROP_FOCUS, counter)

    serial_pulsador = 0

    gyroData = [0, 0, 0, 0]

    # --- Funções reservadas para envio e armazenamento de dados --- #

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

    def checkSerialInput():
        global gyroData, serial_giroscopio
        if (serial_giroscopio.in_waiting > 0):
            ser_line = serial_giroscopio.readline().decode()
            gyroData = [float(x) for x in ser_line.split(",")]  # list  quaternium to biallhole
            serial_giroscopio.flush()

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

        for port in ports:
            print(port)
            if (port.serial_number == "56CA000930") or (port.serial_number == "EC:DA:3B:BF:A7:60"):
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

        print("Testando comunicação serial: encoder")
        _ = serial_pulsador.read()
        print("Testando comunicação serial: giroscópio server")
        _ = serial_giroscopio.read()
        print("Comunicação serial OK")
        print("")

        # -- Inicio da configuração da câmera --- #

        print('Pegando acesso a camera, isso pode demorar um pouco...')
        vid = cv2.VideoCapture(camera_id)

        ret, frame = vid.read()
        img_array = cv2_to_nparray_grayscale(frame)
        if camera_exposure != None:
            print("Definindo exposição da câmera")
            vid.set(cv2.CAP_PROP_EXPOSURE, camera_exposure)

        # Abre o arquivo HTML no navegador padrão
        webbrowser.open('file://' + html_file_path)

        frame_num = -10
        while True:
            try:
                # Comentado para remover serial
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

                    # salvando x e y para o web
                    # Recomendo limitar o salvamento a um intervalo. Mas é só uma sugestão mesmo.

                    totaly = round(total_deltay, 2)
                    totalx = round(total_deltax, 2)

                    r = R.from_quat([gyroData])
                    v = [totalx, totaly, 0]
                    ddd = r.apply(v)

                    lock_quat.acquire()
                    if printar:
                        lista_dados2.append(totalx)
                        lista_dados.append(totaly)
                        lista_imu.append(gyroData)
                        lista_3d.append(ddd)
                    lock_quat.release()

                frame_num = frame_num + 1
                img_processed_old = img_processed

            except KeyboardInterrupt:

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
            except Exception as exc:
                print("Erro:", exc)

    def salvar_dados_arquivo():
        global lista_dados, lista_dados2, lista_imu
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Cria um timestamp único
        arquivo_dados = os.path.join(data_dir, f"dados_{timestamp}.txt")  # Nome do arquivo com timestamp
        with open(arquivo_dados, "w") as arquivo:
            for x, y, z in zip(lista_dados, lista_dados2, lista_imu):
                arquivo.write(
                    f"{x}|{y}|{z}\n")  # Escreve os dados x e y em uma linha, separados por um espaço e com uma quebra de linha no final

    @app.route('/iniciar', methods=["GET", "POST"])
    def iniciar():
        global printar

        printar = True
        return '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;"><form action="/3D" method="post"><button type="submit" style="width: 150px; height: 50px;">3D</button></form><form action="/finalizar" method="post"><button type="submit" style="width: 150px; height: 50px;">stop</button></form></div>'

    @app.route('/3D')
    def vizu3d():
        global lista_3d
        # Dados
        dados = np.array(lista_3d)

        # Extrair valores x, y, z do vetor
        x = dados[:, 0]
        y = dados[:, 1]
        z = dados[:, 2]

        # Passar os dados para o template HTML
        return render_template('index.html', x=x.tolist(), y=y.tolist(), z=z.tolist())

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

    @app.route('/dados', methods=["GET", "POST"])
    def mostrar_dados():
        html = '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;">'

        # Listar todos os arquivos .txt no diretório atual
        arquivos_txt = [arquivo for arquivo in os.listdir(data_dir) if arquivo.endswith(".txt")]

        # Criar botões para cada arquivo .txt
        for arquivo in arquivos_txt:
            html += f'<form action="/abrir_arquivo/{arquivo}" method="post"><button type="submit" style="width: 150px; height: 50px;">{arquivo}</button></form>'

        html += '<br><br><form action="/iniciar" method="post"><button type="submit" style="width: 150px; height: 50px;">start</button></form></div>'
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
                abort(
                    404)  # Retorna um erro 404 se o arquivo não existir  <form action="/download" method="post"><button type="submit" style="width: 150px; height: 50px;">download</button>

        except Exception as e:

            return "Erro interno do servidor", 500  # Retorna um erro 500 se ocorrer uma exceção

    @app.route('/download/<nome_arquivo>', methods=["GET"])
    def download(nome_arquivo):
        try:
            arquivo_path = os.path.join(data_dir, nome_arquivo)
            if os.path.isfile(arquivo_path):
                return send_file(arquivo_path, as_attachment=True)
            else:
                abort(404)  # Retorna um erro 404 se o arquivo não existir
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

    if __name__ == "__main__":
        thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        thread_executor.submit(minha_thread)
        app.run(host="127.0.0.1", port=5000)

def salvar_dados_arquivo():
    global lista_dados, lista_dados2, lista_imu
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Cria um timestamp único
    arquivo_dados = os.path.join(data_dir, f"dados_{timestamp}.txt")  # Nome do arquivo com timestamp
    with open(arquivo_dados, "w") as arquivo:
        for x, y, z in zip(lista_dados, lista_dados2, lista_imu):
            arquivo.write(
                f"{x}|{y}|{z}\n")  # Escreve os dados x e y em uma linha, separados por um espaço e com uma quebra de linha no final


@app.route('/iniciar', methods=["GET", "POST"])
def iniciar():
    global printar

    printar = True
    return '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;"><form action="/3D" method="post"><button type="submit" style="width: 150px; height: 50px;">3D</button></form><form action="/finalizar" method="post"><button type="submit" style="width: 150px; height: 50px;">stop</button></form></div>'

@app.route('/3D')
def vizu3d():
    global lista_3d
    # Dados
    dados = np.array(lista_3d)

    # Extrair valores x, y, z do vetor
    x = dados[:, 0]
    y = dados[:, 1]
    z = dados[:, 2]

    # Passar os dados para o template HTML
    return render_template('index.html', x=x.tolist(), y=y.tolist(), z=z.tolist())


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


@app.route('/dados', methods=["GET", "POST"])
def mostrar_dados():
    html = '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;">'

    # Listar todos os arquivos .txt no diretório atual
    arquivos_txt = [arquivo for arquivo in os.listdir(data_dir) if arquivo.endswith(".txt")]

    # Criar botões para cada arquivo .txt
    for arquivo in arquivos_txt:
        html += f'<form action="/abrir_arquivo/{arquivo}" method="post"><button type="submit" style="width: 150px; height: 50px;">{arquivo}</button></form>'

    html += '<br><br><form action="/iniciar" method="post"><button type="submit" style="width: 150px; height: 50px;">start</button></form></div>'
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
            abort(
                404)  # Retorna um erro 404 se o arquivo não existir  <form action="/download" method="post"><button type="submit" style="width: 150px; height: 50px;">download</button>

    except Exception as e:

        return "Erro interno do servidor", 500  # Retorna um erro 500 se ocorrer uma exceção


@app.route('/download/<nome_arquivo>', methods=["GET"])
def download(nome_arquivo):
    try:
        arquivo_path = os.path.join(data_dir, nome_arquivo)
        if os.path.isfile(arquivo_path):
            return send_file(arquivo_path, as_attachment=True)
        else:
            abort(404)  # Retorna um erro 404 se o arquivo não existir
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



if __name__ == "__main__":
    thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    thread_executor.submit(minha_thread)
    app.run(host="127.0.0.1", port=5000)
