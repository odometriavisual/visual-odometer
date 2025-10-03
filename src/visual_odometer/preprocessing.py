"""

Important pre-processing steps such as windowing and downsampling.

"""

from PIL import Image
from .dsp import *


def apply_spatial_window(img: NDArray, method: str, params: dict) -> NDArray:
    """
    Interface that can apply different types of spatial windows.

    Parameters
    ----------
    img : NDArray
        A 2-D array which represents the image to be windowed.
    method : str
        Which spatial window to be applied.
    params : dict
        Parameters related to the chosen window.

    Returns
    -------
    NDArray
        A 2-D array which represents the windowed image.

    Raises
    ------
    NotImplementedError
        If the ``method`` is not among the implemented spatial or temporal windowing methods.  
    """

    match method:
        case "blackman-harris":
            a0, a1, a2, a3 = params['a0'], params['a1'], params['a2'], params['a3']
            return apply_blackman_harris_window(img, a0, a1, a2, a3).astype(np.float32)
        case "raised-cosine" | "raised_cosine":
            return apply_raised_cosine_window(img).astype(np.float32)
        case "" | None:
            return img
        case _:
            raise NotImplementedError(f'Invalid spatial window method: {method}')


def apply_downsampling(img: NDArray[np.float32], method: str, params: dict) -> NDArray[np.float32]:
    """
    Interface that can apply different types of downsampling algorithms.

    Parameters
    ----------
    img : NDArray[np.float32]
        A 2-D array which represents the image to be downsampled.
    method : str
        Which downsample algorithm to be applied.
    params : dict
        Parameters related to the specific downsample algorithm.

    Returns
    -------
    NDArray[np.float32]
         A 2-D array which represents the downsampled image.

    Raises
    ------
    NotImplementedError
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
            raise NotImplementedError(f"Invalid downsampling method: {method}")


def apply_frequency_window(spectrum: NDArray[np.complex64], method: str, params: dict) -> NDArray[np.complex64]:
    """
    Interface that can apply different types of frequency windows.

    Parameters
    ----------
    spectrum : NDArray[np.complex64]
        A 2-D array which represents the image to be windowed.
    method : str
        Which frequency window to be applied.
    params : dict
        Parameters related to the chosen window.

    Returns
    -------
    NDArray[np.complex64]
        A 2-D array which represents the windowed image.

    Raises
    ------
    NotImplementedError
        If the ``method`` is not among the implemented frequency windowing methods.

    """

    match method:
        case "Stone_et_al_2001" | "ideal-lowpass":
            return ideal_lowpass(spectrum, params["factor"])
        case "" | None:
            return spectrum
        case _:
            raise NotImplementedError(f'Invalid frequency window method: {method}')


def image_preprocessing(img: NDArray[np.float32], configs: dict) -> NDArray[np.complex64]:
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
        params=configs["Downsampling"]["params"]
    )

    # Apply spatial windowing:
    img = apply_spatial_window(
        img,
        method=configs["Spatial Window"]["method"],
        params=configs["Spatial Window"]["params"]
    )

    img_spectrum = np.fft.fftshift(np.fft.fft2(img))
    img_spectrum = apply_frequency_window(
        img_spectrum,
        method=configs["Frequency Window"]["method"],
        params=configs["Frequency Window"]["params"]
    )
    return img_spectrum
