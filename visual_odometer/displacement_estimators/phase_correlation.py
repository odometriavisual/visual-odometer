import numpy as np
from numpy.typing import NDArray


def subpixel_peak_position(corr_abs: NDArray[np.float32], method ="max") -> tuple[float, float]:
    r"""
    Extract from the time-domain 2D correlation :math:`r[y,x]` the displacement value :math:`[\Delta y, \Delta x]^T`,
     assuming the correlation was performed between two shifted images :math:`f[y, x]` and :math:`g[y, x]`:

    .. math::
        g[y, x] = f[y + \Delta y, x + \Delta x]

    .. math::
        R[u, v] = \dfrac{F[u, v] \odot G[u, v]^*}{|F[u, v] \cdot G[u, v]|}

    .. math::
        r[y, x] = \mathcal{F}^{-1}\{ R[u,v] \}[y, x]

    .. math::
        (\Delta y, \Delta x) = \\text{method}\Big( r[y, x] \Big)

    Parameters
    ----------
    corr_abs : NDArray[np.float32]
        A 2-D Array representing :math:`r[y,x]` which is the time-domain correlation matrix between :math:`f[y, x]` and :math:`g[y, x]`.
    method : {"max"}, optional
        Which displacement detection method, by default "max".

    Returns
    -------
    tuple[float, float]
        Horizontal and vertical displacement values :math:`[\Delta y, \Delta x]^T`.

    Raises
    ------
    ValueError
        If the ``method`` is not among the implemented methods.
    """
    mid_y, mid_x = corr_abs.shape[0] // 2, corr_abs.shape[1] // 2

    match method:
        case "max":
            peak_y, peak_x = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
            dx = peak_x - mid_x
            dy = peak_y - mid_y
        case _:
            raise ValueError(f"Invalid peak detection method: {method}")

    return float(dx), float(dy)


def phase_correlation_method(fft_beg: NDArray[np.complex64], fft_end: NDArray[np.complex64], method ='max') -> tuple[float, float]:
    r"""
    Estimate vertical and horizontal displacement vector :math:`[\Delta y, \Delta x]^T` between two spatially shifted spectra:

    .. math::

        I_{end}[y, x] = I_{beg}[y - \Delta y, x - \Delta x]


    where `fft_beg` is :math:`\mathcal{F}\{ I_{beg}[y, x]\} (u, v)` and `fft_end` is :math:`\mathcal{F}\{ I_{end}[y, x]\} (u, v)`.

    The algorithm behind the estimation it is the Phase Correlation (PC) :cite:`foroosh_extension_2002`.

    Parameters
    ----------
    fft_beg : NDArray[np.complex64]
        A 2-D array representing the spectrum of :math:`I_{beg}[y, x]`
    fft_end : NDArray[np.complex64]
        A 2-D array representing the spectrum of :math:`I_{end}[y,x]`
    method : {"max"}, optional
        Method to extract shift value from time-domain correlation matrix, by default 'max'

    Returns
    -------
    tuple[float, float]
        Horizontal and vertical displacement values :math:`[\Delta y, \Delta x]^T`, assuming :math:`I[y, x]`.
        
    References
    ----------
    :cite:`foroosh_extension_2002` Foroosh, H., Zerubia, J. B., & Berthod, M. (2002). Extension of phase correlation to subpixel registration. IEEE transactions on image processing, 11(3), 188-200. :doi:`10.1109/83.988953`
    
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
