import numpy as np
from numpy.typing import NDArray


def subpixel_peak_position(corr_abs: NDArray[np.float32], method: str="max") -> tuple[float, float]:
    """
    Extract from the time-domain 2D correlation the displacement value, assuming the correlation was perfomed between two shifted images.

    Parameters
    ----------
    corr_abs : NDArray[np.float32]
        A 2-D Array represeting the correlation matrix between I1[y, x] and I2[y, x] where I2[y, x] = I1[y - dy, x - dx].
    method : str, optional
        Which displacement detection method, by default "max".

    Returns
    -------
    tuple[float, float]
        Horizontal and vertical (x and y) displacement values, assuming I[y, x].

    Raises
    ------
    NotImplementedError
        If the ``method`` is not among the implemented methods.
    """
    mid_y, mid_x = corr_abs.shape[0] // 2, corr_abs.shape[1] // 2

    match method:
        case "max":
            peak_y, peak_x = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
            dx = peak_x - mid_x
            dy = peak_y - mid_y
        case _:
            raise NotImplementedError(f"Not implemented peak detection method: {method}")

    return float(dx), float(dy)


def phase_correlation_method(fft_beg: NDArray[np.complex64], fft_end: NDArray[np.complex64], method: str='max') -> tuple[float, float]:
    """
    Estimate displacement between two spatialy shifted spectra, i.e.:
    
    I_end[y, x] = I_beg[y - dy, x - dx]
    
    where fft_beg = FFT(I_beg) and fft_end = FFT(I_end), by using Phase Correlation (PC) [1]_.

    Parameters
    ----------
    fft_beg : NDArray[np.complex64]
        A 2-D array represeting the spectrum of I_beg
    fft_end : NDArray[np.complex64]
        A 2-D array represeting the spectrum of I_end
    method : str, optional
        Method to extract shift value from time-domain correlation matrix, by default 'max'

    Returns
    -------
    tuple[float, float]
        Horizontal and vertical (x and y) displacement values, assuming I[y, x].
        
    References
    ----------
    .. [1] Foroosh, H., Zerubia, J. B., & Berthod, M. (2002). Extension of phase correlation to subpixel registration. IEEE transactions on image processing, 11(3), 188-200.
    
    """

    # Cross-power spectrum
    R = fft_end * np.conj(fft_beg)
    R /= np.maximum(np.abs(R), 1e-10)  # evitar divisão por zero

    # Correlation (IFFT)
    corr = np.fft.ifft2(R)
    corr = np.fft.fftshift(corr)

    # Deslocamento
    dx, dy = subpixel_peak_position(np.abs(corr), method)

    return dx, dy
