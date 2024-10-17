import numpy as np
from numpy import ndarray

from scipy.ndimage import convolve
from scipy.sparse.linalg import svds

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def normalize_product(F: ndarray, G: ndarray, method="Stone_et_al_2001") -> ndarray:
    # Versão modificada de crosspower_spectrum() para melhorias de eficiência

    Q = F * np.conj(G) / np.abs(F * np.conj(G))
    return Q

def phase_fringe_filter(cross_power_spectrum: ndarray, window_size=(5, 5), threshold=0.03):
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

def svd_method(fft_beg: ndarray, fft_end: ndarray, M: int, N: int, phase_windowing=None, finge_filter=True, use_gpu = False):
    Q = normalize_product(fft_beg, fft_end)
    if finge_filter is True:
        Q = phase_fringe_filter(Q)

    if use_gpu:
        if GPU_AVAILABLE:
            # Usar SVD de Cupy
            qu, s, qv = cp.linalg.svd(Q, full_matrices=False)
            # Obter o ângulo dos vetores U e V
            ang_qu = cp.angle(qu[:, 0])
            ang_qv = cp.angle(qv[0, :])
        else:
            raise Exception("Erro, cupy não está instalado, coloque use_gpu como False ou instale o cupy usando pip install cupy-cuda11x")
    else:
        # Usar SVD de CPU (SciPy)
        qu, s, qv = svds(Q, k=1)
        ang_qu = np.angle(qu[:, 0])
        ang_qv = np.angle(qv[0, :])

    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay = svd_estimate_shift(ang_qu, M, phase_windowing)
    deltax = svd_estimate_shift(ang_qv, N, phase_windowing)

    # round() pois o retorn é em pixels
    return deltax, deltay
