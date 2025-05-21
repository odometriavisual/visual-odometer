import numpy as np
import json
import threading

from .displacement_estimators.svd import svd_method
from .preprocessing import image_preprocessing

DEFAULT_CONFIG = {
    "Displacement Estimation": {
        "method": "svd",
        "params": {}
    },
    "Frequency Window": {
        "method": "Stone_et_al_2001",
        "params": {
            "factor": 0.6,
        }
    },
    "Spatial Window": {
        "method": "blackman_harris",
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
    def __init__(self, img_size: (int, int), xres: float = 1.0, yres: float = 1.0):
        # Default configs:
        self.configs = DEFAULT_CONFIG

        self.img_size = img_size
        self.xres, self.yres = xres, yres  # Relationship between displacement in pixels and millimeters

        self.current_position = np.array([0, 0])  # In pixels
        self.number_of_displacements = 0

        self.imgs_lock = threading.Lock()
        self.imgs_processed = [None, None]
        # The first img in imgs_processed will always be the last successful image used on a displacement estimation.
        # The second img will be the most recent image

    def calibrate(self, new_xres: float, new_yres: float):
        self.xres, self.yres = new_xres, new_yres

    def estimate_displacement_between(self, img_beg: np.ndarray, img_end: np.ndarray) -> (float, float):
        """
        Estimates the displacement between img_beg and img_end.

        Intendend for single shot usage, for estimating displacements between sequences of images use estimate_last_displacement().
        """
        fft_beg = image_preprocessing(img_beg, self.configs)
        fft_end = image_preprocessing(img_end, self.configs)
        return self._estimate_displacement(fft_beg, fft_end)

    def _estimate_displacement(self, fft_beg, fft_end) -> (float, float):
        method = self.configs["Displacement Estimation"]["method"]

        if method == "svd":
            _deltax, _deltay = svd_method(fft_beg, fft_end, self.img_size[1], self.img_size[0])  # In pixels
        elif method == "phase-correlation":
            raise NotImplementedError
        else:
            raise NotImplementedError

        # Convert from pixels to millimeters (or equivalent):
        deltax, deltay = _deltax * self.xres, _deltay * self.yres
        self.current_position = np.array([self.current_position[0] + deltax, self.current_position[1] + deltay])
        return deltax, deltay

    def get_displacement(self):
        try:
            if None is not self.imgs_processed[0] and None is not self.imgs_processed[1]:
                # Compute the displacement:
                spectrum_beg = self.imgs_processed[0]

                with self.imgs_lock:
                    spectrum_end = self.imgs_processed[1].copy()
                    # Update the image buffer:

                self.imgs_processed[0] = spectrum_end

                displacement = self._estimate_displacement(spectrum_beg, spectrum_end)

                # Update the current position:
                self.current_position[0] += displacement[0]
                self.current_position[1] += displacement[1]

                self.number_of_displacements += 1

                return displacement
            else:
                return 0, 0
        except NotImplementedError:
            return None, None

    def feed_image(self, img: np.ndarray) -> None:
        # Update the latest image:
        img_spectrum = image_preprocessing(img, self.configs)

        if self.imgs_processed[0] is None:
            # The first iteration
            self.imgs_processed[0] = img_spectrum
        else:
            # Update the current image:
            new_img = img_spectrum
            with self.imgs_lock:
                self.imgs_processed[1] = new_img

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
