import numpy as np
import json
import threading

from .displacement_estimators import svd_method
from .displacement_estimators import phase_correlation_method
from .displacement_estimators import proj_svd_method
from .displacement_estimators import phase_amplified_correlation_method
from .preprocessing import image_preprocessing
from .dsp import crop_two_imgs_with_displacement

DEFAULT_CONFIG = {
    "Displacement Estimation": {
        "method": "svd",
        "reprocess_displacement": False,
        "skip_frames": False,
        "params": {
            "skip_frames_threshold": 5,
            "reprocess_displacement_count": 1
        },

    },
    "Frequency Window": {
        "method": "Stone_et_al_2001",
        "params": {
            "factor": 0.6,
        }
    },
    "Spatial Window": {
        "method": "raised_cosine",
        "params": {
            "a0": 0.358,
            "a1": 0.47,
            "a2": 0.135,
            "a3": 0.037,
        }
    },
    "Downsampling": {
        "method": "",
        "params": {
            "factor": 1,
        }
    },
}


class VisualOdometer:
    """
    The class implementing the visual odometer.

    The visual odometer works both with the CPU and CUDA GPUS:
        To work with the CPU pass the image as numpy arrays.
        To work with the GPU pass the image as cupy arrays.
        The API remains the same for both cases.

    The visual odometer is capable of woking in the "Single Shot" mode and in the "Sequential" mode
    In the "Single Shot" mode, the visual odometer outputs the displacement between a pair of images.
    In the "Sequential" mode, the visual odometer outputs a stream of N-1 displacements from a sequence of N images.
    """

    def __init__(self, img_shape: (int, int), xres: float = 1.0, yres: float = 1.0):
        """
        Instantiates a visual odometer
        :param img_shape: The shape of the image array as defined by the numpy.ndarray.shape
        :param xres: Ratio of mm/pixels in the x dimension
        :param yres: Ratio of mm/pixels in the y dimension
        """
        # Default configs:
        self.configs = DEFAULT_CONFIG

        self.img_size = img_shape
        self.xres, self.yres = xres, yres  # Relationship between displacement in pixels and millimeters

        self.current_position = np.array([0, 0])  # In pixels
        self.number_of_displacements = 0

        self.imgs_lock = threading.Lock()
        self.imgs_processed = [None, None]
        self.imgs_original = [None, None]

        # The first img in imgs_processed will always be the last successful image used on a displacement estimation.
        # The second img will be the most recent image

    def calibrate(self, new_xres: float, new_yres: float):
        """
        Changes the visual odometer's ratio of mm/pixels
        :param new_xres: New ratio in the x dimension
        :param new_yres: New ration in the y dimension
        """
        self.xres, self.yres = new_xres, new_yres

    def estimate_displacement_between(self, img_beg, img_end) -> (float, float):
        """
        Estimates the displacement between two images

        Intended for the "Single Shot" mode, for estimating displacements between sequences of images use `estimate_last_displacement()`.

        :param img_beg: Image at t = t_0
        :param img_end: Image at t = t₀ + Δt
        :return: x and y displacements in mm
        """

        img_x_size = img_beg.shape[1]
        img_y_size = img_end.shape[0]

        fft_beg = image_preprocessing(img_beg, self.configs)
        fft_end = image_preprocessing(img_end, self.configs)
        return self._estimate_displacement(fft_beg, fft_end, img_x_size, img_y_size)

    def _estimate_displacement(self, fft_beg, fft_end, img_size_x=None, img_size_y=None) -> (float, float):
        method = self.configs["Displacement Estimation"]["method"]
        params = self.configs["Displacement Estimation"]["params"]

        if img_size_x is None:
            img_size_x = self.img_size[1]
            img_size_y = self.img_size[0]

        match method:
            case "svd":
                _deltax, _deltay = svd_method(fft_beg, fft_end, img_size_x, img_size_y, phase_windowing="central")  # In pixels
            case "phase-correlation":
                _deltax, _deltay = phase_correlation_method(fft_beg, fft_end)
            case "projection-svd":
                _deltax, _deltay = proj_svd_method(fft_beg, fft_end, img_size_x, img_size_y, dx_max=30, dy_max=30, phase_windowing="central")
            case "phase-amplified-correlation":
                _deltax, _deltay = phase_amplified_correlation_method(fft_beg, fft_end, gain=3)
            case _:
                raise ValueError(f"Displacement estimation method {method} not valid.")

        # Convert from pixels to millimeters (or equivalent):
        deltax, deltay = _deltax * self.xres, _deltay * self.yres
        self.current_position = np.array([self.current_position[0] + deltax, self.current_position[1] + deltay])
        return deltax, deltay

    def get_displacement(self):
        """
        Get the next displacement in "Sequential" mode.

        :return: Next x and y displacements in mm
        """
        try:
            reprocess_displacement = self.configs["Displacement Estimation"]["reprocess_displacement"]
            skip_frames = self.configs["Displacement Estimation"]["skip_frames"]

            if self.imgs_processed[0] is not None and self.imgs_processed[1] is not None:
                spectrum_beg = self.imgs_processed[0]
                original_img_beg = self.imgs_original[0]

                with self.imgs_lock:
                    spectrum_end = self.imgs_processed[1].copy()
                    original_img_end = self.imgs_original[1].copy()

                # Estimar deslocamento bruto
                displacement = self._estimate_displacement(spectrum_beg, spectrum_end)
                if reprocess_displacement:
                    count = self.configs["Displacement Estimation"]["params"].get("reprocess_displacement_count", 1)
                    for _ in range(count):
                        round_dx = int(round(displacement[0]))
                        round_dy = int(round(displacement[1]))
                        crop_img_beg, crop_img_end = crop_two_imgs_with_displacement(original_img_beg, original_img_end,
                                                                                     round_dx, round_dy)
                        new_displacement = self.estimate_displacement_between(crop_img_beg, crop_img_end)
                        displacement = [round_dx + new_displacement[0], round_dy + new_displacement[1]]

                if skip_frames:
                    threshold = self.configs["Displacement Estimation"]["params"]["skip_frames_threshold"]
                    if np.linalg.norm(displacement) < threshold:
                        # Não atualiza a imagem base (mantém img_beg)
                        return 0.0, 0.0

                # Atualiza img base apenas se deslocamento foi aceito
                self.imgs_processed[0] = spectrum_end
                self.imgs_original[0] = original_img_end

                self.current_position[0] += displacement[0]
                self.current_position[1] += displacement[1]
                self.number_of_displacements += 1

                return displacement
            else:
                return 0.0, 0.0
        except NotImplementedError:
            return None, None

    def feed_image(self, img) -> None:
        """
        Send the next image in "Sequential" mode for processing
        :param img: Next image in the stream
        """

        # Update the latest image:
        img_spectrum = image_preprocessing(img, self.configs)

        if self.imgs_processed[0] is None:
            # The first iteration
            self.imgs_processed[0] = img_spectrum
            self.imgs_original[0] = img
        else:
            # Update the current image:
            new_img = img_spectrum
            with self.imgs_lock:
                self.imgs_processed[1] = new_img
                self.imgs_original[1] = img

    def _config(self, arg1: str, arg2: str, arg3: dict):
        self.configs[arg1]["method"] = arg2
        self.configs[arg1]["params"] = arg3

    def config_displacement_estimation(self, method: str = "", **kwargs):
        self._config("Displacement Estimation", method, kwargs)

    def config_frequency_window(self, method: str = "", **kwargs):
        self._config("Frequency Window", method, kwargs)

    def config_spatial_window(self, method: str = "", **kwargs):
        self._config("Spatial Window", method, kwargs)

    def config_downsampling(self, method: str = "", **kwargs):
        self._config("Downsampling", method, kwargs)

    def set_config(self, new_config):
        self.configs = new_config

    def print_config(self):
        print(self.configs)

    def save_config(self, path: str, filename="visual-odometer-config"):
        try:
            with open(path + "/" + filename + ".json", 'w') as fp:
                json.dump(self.configs, fp)
            return True
        except:
            return False
