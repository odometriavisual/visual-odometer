# other_operations.py

import numpy as np
import scipy.ndimage

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
