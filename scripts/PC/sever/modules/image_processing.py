# image_processing.py

import numpy as np
import scipy.ndimage
import scipy.signal

def normalize_product(F, G, method="Stone_et_al_2001"):
    # Versão modificada de crosspower_spectrum() para melhorias de eficiência

    Q = F * np.conj(G) / np.abs(F * np.conj(G))
    return Q

def apply_window(img1, img2, window_type):
    window = generate_window(img1.shape, window_type)
    return img1 * window, img2 * window

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
