import numpy as np
from numpy.fft import fft2, fftshift
import cv2
from PIL import Image, ImageOps

from virtualencoder.visualodometry.image_utils import apply_blackman_harris_window


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
