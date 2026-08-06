"""

Important pre-processing steps such as windowing and downsampling.

"""

from PIL import Image
import numpy as np
from numpy.typing import NDArray

from visual_odometer import dsp

def apply_spatial_window(img: NDArray, method, params: dict) -> NDArray:
    """
    Interface that can apply different types of spatial windows.

    Parameters
    ----------
    img : NDArray
        A 2-D array which represents the image to be windowed.
    method : {"blackman-harris", "raised-cosine", None}
        Which spatial window to be applied.
    params : dict
        Parameters related to the chosen window.

    Returns
    -------
    NDArray
        A 2-D array which represents the windowed image.

    Raises
    ------
    ValueError
        If the ``method`` is not among the implemented spatial or temporal windowing methods.
    """

    match method:
        case "blackman-harris":
            a0, a1, a2, a3 = params["a0"], params["a1"], params["a2"], params["a3"]
            return dsp.apply_blackman_harris_window(img, a0, a1, a2, a3).astype(
                np.float32
            )
        case "raised-cosine" | "raised_cosine":
            return dsp.apply_raised_cosine_window(img).astype(np.float32)
        case "" | None:
            return img
        case _:
            raise ValueError(f"Invalid spatial window method: {method}")


def apply_downsampling(
    img: NDArray[np.float32], method, params: dict
) -> NDArray[np.float32]:
    """
    Interface that can apply different types of downsampling algorithms.

    Parameters
    ----------
    img : NDArray[np.float32]
        A 2-D array which represents the image to be downsampled.
    method :  {“NN”, “bilinear”, "bicubic", None}
        Which downsample algorithm to be applied.
    params : dict
        Parameters related to the specific downsample algorithm.

    Returns
    -------
    NDArray[np.float32]
         A 2-D array which represents the downsampled image.

    Raises
    ------
    ValueError
        If the ``method`` is not among the implemented downsampled algorithms.
    """
    factor = params["factor"]
    newsize = int(img.shape[0] / factor), int(img.shape[1] / factor)
    img_pil = Image.fromarray(img)

    match method:
        case "NN":
            return np.array(img_pil.resize(newsize, Image.NEAREST))
        case "bilinear":
            return np.array(img_pil.resize(newsize, Image.BILINEAR))
        case "bicubic":
            return np.array(img_pil.resize(newsize, Image.BICUBIC))
        case "" | None:
            return img
        case _:
            raise ValueError(f"Invalid downsampling method: {method}")


def apply_frequency_window(
    spectrum: NDArray[np.complex64], method, params: dict
) -> NDArray[np.complex64]:
    """
    Interface that can apply different types of frequency windows.

    Parameters
    ----------
    spectrum : NDArray[np.complex64]
        A 2-D array which represents the image to be windowed.
    method : {“Stone_et_al_2001”, “ideal-lowpass”, None}
        Which frequency window to be applied.
    params : dict
        Parameters related to the chosen window.

    Returns
    -------
    NDArray[np.complex64]
        A 2-D array which represents the windowed image.

    Raises
    ------
    ValueError
        If the ``method`` is not among the implemented frequency windowing methods.

    """

    match method:
        case "Stone_et_al_2001" | "ideal-lowpass":
            return dsp.ideal_lowpass(spectrum, params["factor"])
        case "" | None:
            return spectrum
        case _:
            raise ValueError(f"Invalid frequency window method: {method}")


def phase_unwrap(
    phase_wrapped: NDArray[np.float32], method="itoh1982"
) -> NDArray[np.float32]:
    r"""
    Interface for applying different types of phase unwrapping algorithms.

    Parameters
    ----------
    phase_wrapped : NDArray[np.float32]
         A 1-D Array representing the wrapped phase values that is limited to the :math:`]-\pi, +\pi]` interval.
    method : {"itoh1982", "numpy"}, optional
        Phase unwrapping method, by default "itoh1982".

    Returns
    -------dsadas
    NDArray[np.float32]
        A 1-D Array representing unwrapped phase values that could range from :math::math:`]-\infty, +\infty]`.

    Raises
    ------
    ValueError
        If the ``method`` is not among the implemented phase unwrapping methods.
    """

    match method:
        case "itoh1982":
            return itoh1982_method(phase_wrapped)
        case "numpy":
            return np.unwrap(phase_wrapped)
        case _:
            raise ValueError(f"Phase unwrap method {method} not valid.")


def itoh1982_method(
    phase_vec: NDArray[np.float32], factor: float = 0.7
) -> NDArray[np.float32]:
    r"""
    Phase unwrapping method based on :cite:`itoh_analysis_1982`.

    Parameters
    ----------
    phase_vec : NDArray[np.float32]
         A 1-D Array representing the wrapped phase values that is limited to the :math:`]-\pi, +\pi]` interval.
    factor : float, optional
        A constant that defines how close the first-order difference between two consecutive phase samples must be to :math:`2\pi` to be considered a wrapping event, by default 0.7

    Returns
    -------
    NDArray[np.float32]
        A 1-D Array representing unwrapped phase values that could range from :math::math:`]-\infty, +\infty]`.

    References
    ----------
    :cite:`itoh_analysis_1982` Itoh, K. (1982). Analysis of the phase unwrapping algorithm. Applied optics, 21(14), 2470-2470. :doi:`10.1364/AO.21.002470`
    """
    phase_diff = np.diff(phase_vec)
    corrected_difference = (
        phase_diff
        - 2.0 * np.pi * (phase_diff > (2 * np.pi * factor))
        + 2.0 * np.pi * (phase_diff < -(2 * np.pi * factor))
    )
    return np.cumsum(corrected_difference)


def pc_analyze_image(img: NDArray[np.float32], configs: dict) -> NDArray[np.complex64]:
    """
    Function that applies a pipeline of image-processing steps.

    Parameters
    ----------
    img : NDArray[np.float32]
         A 2-D Array that represents a grey-scale image.
    configs : dict
        Set of configurations to the pre-processing steps.

    Returns
    -------
    NDArray[np.complex64]
        A 2-D Array that represents the spectrum of ``img`` after an image processing pipeline.
    """

    # Function which applies all the preprocessing
    # Apply downsampling:
    img = apply_downsampling(
        img,
        method=configs["Downsampling"]["method"],
        params=configs["Downsampling"]["params"],
    )

    # Apply spatial windowing:
    img = apply_spatial_window(
        img,
        method=configs["Spatial Window"]["method"],
        params=configs["Spatial Window"]["params"],
    )

    img_spectrum = np.fft.fftshift(np.fft.fft2(img))
    img_spectrum = apply_frequency_window(
        img_spectrum,
        method=configs["Frequency Window"]["method"],
        params=configs["Frequency Window"]["params"],
    )
    return img_spectrum
