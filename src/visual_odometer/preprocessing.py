from PIL import Image
from .dsp import *

try:
    import cupy as cp
except:
    cp = None

def apply_spatial_window(img, method: str, params: dict, use_gpu=False):
    if method == "blackman_harris":
        return apply_blackman_harris_window(img, params['a0'], params['a1'], params['a2'], params['a3'], use_gpu=use_gpu).astype(np.float32)
    elif method == "raised_cosine":
        return apply_raised_cosine_window(img, use_gpu).astype(np.float32)
    elif method == "" or method == None:
        return img
    else:
        print(f'Atenção: tentando aplicar o método de janelamento de imagem {method}, mas ele não está implementado.')
        return img

def apply_downsampling(img, method: str, params: dict, use_gpu=False):
    if method == "" or method == None:
        return img
    elif use_gpu:
        print("Downsampling on gpu is not implemented")
        return img

    factor = params["factor"]
    newsize = int(img.shape[0] / factor), int(img.shape[1] / factor)
    img_pil = Image.fromarray(img)

    if method == "NN":
        return np.array(img_pil.resize(newsize, Image.NEAREST))
    elif method == "bilinear":
        return np.array(img_pil.resize(newsize, Image.BILINEAR))
    elif method == "bicubic":
        return np.array(img_pil.resize(newsize, Image.BICUBIC))
    else:
        return img

def apply_frequency_window(spectrum: np.ndarray, method: str, params: dict):
    if method == "Stone_et_al_2001":
        return ideal_lowpass(spectrum, params["factor"])
    elif method == "":
        return spectrum
    else:
        print(f'Atenção: tentando aplicar o método {method}, mas ele não está implementado.')
        return spectrum

def image_preprocessing(img, configs: dict, use_gpu = False):
    # Function which applies all the preprocessing
    # Apply downsampling:
    img = apply_downsampling(
        img,
        method=configs["Downsampling"]["method"],
        params=configs["Downsampling"]["params"],
        use_gpu=use_gpu
    )

    #Apply spatial windowing:
    img = apply_spatial_window(
        img,
        method=configs["Spatial Window"]["method"],
        params=configs["Spatial Window"]["params"],
        use_gpu=use_gpu
    )

    if use_gpu is True:
        img_spectrum = cp.fft.fftshift(cp.fft.fft2(img))
        img_spectrum = apply_frequency_window(
            img_spectrum,
            method=configs["Frequency Window"]["method"],
            params=configs["Frequency Window"]["params"]
        )
    else:
        img_spectrum = np.fft.fftshift(np.fft.fft2(img))
        img_spectrum = apply_frequency_window(
            img_spectrum,
            method=configs["Frequency Window"]["method"],
            params=configs["Frequency Window"]["params"]
        )
    return img_spectrum
