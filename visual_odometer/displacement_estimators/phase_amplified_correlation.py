import numpy as np
from numpy.typing import NDArray
from .phase_correlation import subpixel_peak_position


def phase_amplified_correlation_method(fft_beg: NDArray[np.complex64], fft_end: NDArray[np.complex64], gain:int =0, method ='max') -> tuple[float, float]:
    r"""
    Estimate vertical and horizontal displacement vector :math:`[\Delta y, \Delta x]^T` between two spatially shifted spectra:

    .. math::

        I_{end}[y, x] = I_{beg}[y - \Delta y, x - \Delta x]

    by using the classic phase-correlation method :cite:`itoh_analysis_1982` to a new image :math:`I_{end}'[y, x]` with a
    gain applied over the displacements:

    .. math::

        I_{end}'[y, x] = I_{beg}[y - m\Delta y, x - m\Delta x]
    
    where `fft_beg` is :math:`\mathcal{F}\{ I_{beg}[y, x]\} (u, v)` and `fft_end` is :math:`\mathcal{F}\{ I_{end}[y, x]\} (u, v)` and
    :math:`m` is a user-defined gain.

    The algorithm behind the estimation it is the Phase Amplified Correlation (PAC) :cite:`konstantinidis_phase_2019`.

    Parameters
    ----------
    fft_beg : NDArray[np.complex64]
        A 2-D array representing the spectrum of :math:`I_{beg}[y, x]`
    fft_end : NDArray[np.complex64]
        A 2-D array representing the spectrum of :math:`I_{end}[y,x]`
    gain : int, optional
        The phase gain, by default 0, yielding the classical phase-correlation method.
    method : {"max"}, optional
        Method to extract shift value from time-domain correlation matrix, by default 'max'

    Returns
    -------
    tuple[float, float]
         Horizontal and vertical (x and y) displacement values, assuming :math:`I[y, x]`.
         
    References
    ----------
     :cite:`konstantinidis_phase_2019` Konstantinidis, D., Stathaki, T., & Argyriou, V. (2019). Phase amplified correlation for improved sub-pixel motion estimation. IEEE Transactions on Image Processing, 28(6), 3089-3101. :doi:`10.1109/TIP.2019.2894266`.

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
