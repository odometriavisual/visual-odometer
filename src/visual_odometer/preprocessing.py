import numpy as np
from numpy.fft import fft2, fftshift
from PIL import Image
from .dsp import *


def apply_spatial_window(img: ndarray, method: str, params: dict):
    match method:
        case "blackman_harris":
            return apply_blackman_harris_window(img, params['a0'], params['a1'], params['a2'], params['a3'])
        case "raised_cosine":
            return apply_raised_cosine_window(img)
        case "":
            return img
        case _:
            print(f'Atenção: tentando aplicar o método de janelamento de imagem {method}, mas ele não está implementado.')
            return img


def apply_downsampling(img: np.ndarray, method: str, params: dict):
    factor = params["factor"]
    newsize = int(img.shape[0] / factor), int(img.shape[1] / factor)
    img_pil = Image.fromarray(img)

    match method:
        case "NN":
            return np.array(img_pil.resize(newsize, Image.NEAREST))
        case "bilinear":
            return np.array(img_pil.resize(newsize, Image.BILINEAR))
        case "bicubic":
            return np.array(img_pil.resize(newsize, Image.BICUBIC))
        case "":
            return img
        case _:
            print(f'Atenção: tentando aplicar o método de downsampling {method}, mas ele não está implementado.')
            return img


def apply_frequency_window(spectrum: np.ndarray, method: str, params: dict):
    match method:
        case "Stone_et_al_2001":
            return ideal_lowpass(spectrum, params["factor"])
        case "":
            return spectrum
        case _:
            print(f'Atenção: tentando aplicar o método {method}, mas ele não está implementado.')
            return spectrum


# Function which applies all the preprocessing:

def image_preprocessing(img: np.ndarray, configs: dict):
    # Apply downsampling:
    img = apply_downsampling(
        img,
        method=configs["Downsampling"]["method"],
        params=configs["Downsampling"]["params"]
    )

    # Apply spatial windowing:
    img = apply_spatial_window(
        img,
        method=configs["Spatial Window"]["method"],
        params=configs["Spatial Window"]["params"]
    )

    # Apply FFT:
    img_spectrum = fftshift(fft2(img))
    img_spectrum = apply_frequency_window(
        img_spectrum,
        method=configs["Frequency Window"]["method"],
        params=configs["Frequency Window"]["params"]
    )

    return img_spectrum
