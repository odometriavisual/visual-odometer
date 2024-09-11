import numpy as np
from numpy.fft import fft2, fftshift

import cv2
from PIL import Image, ImageOps

def cv2_to_nparray_grayscale(frame):
    cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_array_rgb = Image.fromarray(cv2_img)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    img_array = np.asarray(img_grayscale)
    return img_array


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
    fft_from_image = fftshift(fft2(image))
    if method is not None:
        fft_from_image = ideal_lowpass(fft_from_image, method=method)
    return fft_from_image