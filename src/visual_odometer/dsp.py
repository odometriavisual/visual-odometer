"""

General Digital Signal Processing (DSP) utilities.

"""

from numpy.typing import NDArray
import numpy as np


# Frequency Windows:

def ideal_lowpass(img: NDArray, factor: float = 0.6) -> NDArray:
    """
    Ideal low-pass filter assuming 'img' is fftshifted, i.e. the zero frequency is at the middle of the spectra.

    Parameters
    ----------
    img : NDArray
        A 2-D Array representing the image to be filtered
    factor : float, optional
        How much to be kept from the original image, by default 0.6 or 60%

    Returns
    -------
    NDArray
        A 2-D array representing the filtered image with reduced size.
        
    """
    
    m = factor * img.shape[0] / 2
    n = factor * img.shape[1] / 2

    N = np.min(np.array([m, n]))
    N_val = int(N)
    img = img[int(img.shape[0] // 2 - N_val): int(img.shape[0] // 2 + N_val),
        int(img.shape[1] // 2 - N_val): int(img.shape[1] // 2 + N_val)]

    return img


# Spatial Windows:
def apply_raised_cosine_window(img: NDArray) -> NDArray:
    """
    Raised-cosined window.

    Parameters
    ----------
    img : NDArray
        A 2-D Array representing the image to apply the window.

    Returns
    -------
    NDArray
        A 2-D Array representing the image after windowing.
    """

    rows, cols = img.shape

    i = np.arange(rows)
    j = np.arange(cols)
    window = 0.5 * (1 + np.cos(np.pi * (2 * i[:, None] - rows) / rows)) * \
             0.5 * (1 + np.cos(np.pi * (2 * j - cols) / cols))
    return img * window


def blackman_harris_window(size: int, a0: float, a1: float, a2: float, a3: float) -> NDArray:
    """
    General formulation for a Blackman-Harris window, i.e. 
    
    w(n) = a0 - a1 · cos(2πn / (N - 1)) + a2 · cos(4πn / (N - 1)) - a3 · cos(6πn / (N - 1)),
    
    where:
    - n is the sample index: n = 0, 1, ..., N - 1
    - N is the total number of samples in the window
    - a0, a1, a2, a3 are real-valued coefficients 
    
    Parameters
    ----------
    size : int
        Window size.
    a0 : float
        Parameter a0.
    a1 : float
        a1.
    a2 : float
        a2.
    a3 : float
        a3.

    Returns
    -------
    NDArray
        A 1-D array of non-zero elements of the window.
    """

    n = np.arange(size)
    window = (a0
              - a1 * np.cos(2 * np.pi * n / (size - 1))
              + a2 * np.cos(4 * np.pi * n / (size - 1))
              - a3 * np.cos(6 * np.pi * n / (size - 1)))
    return window


def apply_blackman_harris_window(img: NDArray,
                                 a0: float = 0.35875, a1: float = 0.48829,
                                 a2: float = 0.14128, a3: float = 0.01168) -> NDArray:
    """
    Interface to apply the Blackman-Harris window.

    Parameters
    ----------
    img : NDArray
        2-D array representing image to be windowed.
    a0 : float, optional
         a0 parameter for the Blackman-Harris window, by default 0.35875
    a1 : float, optional
        a1 parameter for the Blackman-Harris window, by default 0.48829
    a2 : float, optional
        a2 parameter for the Blackman-Harris window, by default 0.14128
    a3 : float, optional
        a3 parameter for the Blackman-Harris window, by default 0.01168

    Returns
    -------
    NDArray
        2-D array representing windowed image
    """

    height, width = img.shape
    window_row = blackman_harris_window(width, a0, a1, a2, a3)
    window_col = blackman_harris_window(height, a0, a1, a2, a3)
    img_windowed = np.outer(window_col, window_row) * img
    return img_windowed


def crop_two_imgs_with_displacement(imgA: NDArray, imgB: NDArray, dx: float, dy: float) -> NDArray:
    """
    Crops two image to preserve the intersection region, assuming imgA is imgB spatially shifted by dx and dy, i.e. imgA[y, x] = imgB[y - dy, x - dx].

    Parameters
    ----------
    imgA : NDArray
        A 2-D array representing an image.
    imgB : NDArray
        A 2-D array representing an image.
    dx : float
        The horizontal shift (x-axis) between imgA and imgB.
    dy : float
        The vertical shift (y-axis) between imgA and imgB.

    Returns
    -------
    NDArray
        A 2-D array representing the intersection region between imgA and imgB.
    """

    h, w = imgA.shape

    # Corte no eixo x (invertido)
    if dx > 0:
        imgA = imgA[:, :w - dx]
        imgB = imgB[:, dx:]
    elif dx < 0:
        dx = abs(dx)
        imgA = imgA[:, dx:]
        imgB = imgB[:, :w - dx]

    # Corte no eixo y (invertido)
    if dy > 0:
        imgA = imgA[:h - dy, :]
        imgB = imgB[dy:, :]
    elif dy < 0:
        dy = abs(dy)
        imgA = imgA[dy:, :]
        imgB = imgB[:h - dy, :]

    # Garante que as imgns finais tenham o mesmo tamanho
    min_h = min(imgA.shape[0], imgB.shape[0])
    min_w = min(imgA.shape[1], imgB.shape[1])
    imgA = imgA[:min_h, :min_w]
    imgB = imgB[:min_h, :min_w]

    return imgA, imgB


# Normalized cross-power spectrum:
def normalized_cps(F: NDArray[np.complex64], G: NDArray[np.complex64], epsilon: float = 1e-10) -> NDArray[np.complex64]:
    """
    Normalized Cross Power Spectrum (CPS). F and G are two same size spectra, thus both are complex matrices.

    Parameters
    ----------
    F : NDArray[np.complex64]
        A 2-D complex-valued array representing the spectra of an image f(x, y)
    G : NDArray[np.complex64]
        A 2-D complex-valued array representing the spectra of an image g(x, y)
    epsilon : float, optional
        Constant to prevent division by zero, by default 1e-10

    Returns
    -------
    NDArray[np.complex64]
        Cross-power spectrum between spectra of f(x, y) and g(x, y)
    """
    
    Q = F * np.conj(G)
    Q /= np.abs(Q) + epsilon  # avoid zero-division
    return Q

#
