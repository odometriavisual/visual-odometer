import numpy as np
from numpy.typing import NDArray
from .phase_correlation import subpixel_peak_position


def phase_amplified_correlation_method(fft_beg: NDArray[np.complex64], fft_end: NDArray[np.complex64], gain:int =0, method:str ='max') -> tuple[float, float]:
    """
    Estimate displacement between two spatialy shifted spectra, i.e.:
    
    I_end[y, x] = I_beg[y - dy, x - dx]
    
    where fft_beg = FFT(I_beg) and fft_end = FFT(I_end), by using Phase Amplified Correlation (PAC) [1]_.

    Parameters
    ----------
    fft_beg : NDArray[np.complex64]
        A 2-D array represeting the spectrum of I_beg
    fft_end : NDArray[np.complex64]
        A 2-D array represeting the spectrum of I_end
    gain : int, optional
        The phase gain, by default 0
    method : str, optional
        Method to extract shift value from time-domain correlation matrix, by default 'max'

    Returns
    -------
    tuple[float, float]
         Horizontal and vertical (x and y) displacement values, assuming I[y, x].
         
    References
    ----------
    .. [1] Konstantinidis, D., Stathaki, T., & Argyriou, V. (2019). Phase amplified correlation for improved sub-pixel motion estimation. IEEE Transactions on Image Processing, 28(6), 3089-3101.
    
    """    
    
    # Regular cross-power spectrum
    R = fft_end * np.conj(fft_beg)
    R /= np.maximum(np.abs(R), 1e-10)  # evitar divisão por zero

    # Amplification step:
    R_amplified = R ** (1 + gain)

    # Correlation (IFFT)
    corr = np.fft.ifft2(R_amplified)
    corr = np.fft.fftshift(corr)

    # Deslocamento
    dx_, dy_ = subpixel_peak_position(np.abs(corr), method)

    # Correct the gain:
    dx, dy = dx_ / (1 + gain), dy_ / (1 + gain)

    return dx, dy
