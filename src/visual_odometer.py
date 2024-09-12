import numpy as np
import json
from scipy.sparse.linalg import svds
from src.displacement_estimators import svd_method
from numpy.fft import fftshift, fft2

from src.preprocessing import image_preprocessing


class VisualOdometer:
    def __init__(self, displacement_algorithm="svd", frequency_window="Stone_et_al_2001",
                 spatial_window="blackman-harris", img_size=(640, 480), xres=1, yres=1):
        self.displacement_algorithm = displacement_algorithm
        self.frequency_window = frequency_window
        self.spatial_window = spatial_window
        self.img_size = img_size
        self.xres, self.yres = xres, yres  # Relationship between displacement in pixels and millimeters
        self.current_position = (0, 0)
        self.img_processed = list()

    def calibrate(self, new_xres: float, new_yres: float):
        self.xres, self.yres = new_xres, new_yres

    def get_distance(self, img_beg: np.ndarray, img_end: np.ndarray):
        fft_beg = image_preprocessing(img_beg)  # dessa forma é sempre feito o preprocessamento EM DOBRO! (Duas vezes na mesma imagem)
        fft_end = image_preprocessing(img_end)  # dessa forma é sempre feito o preprocessamento EM DOBRO!

        if self.displacement_algorithm == "svd":
            _deltax, _deltay = svd_method(fft_beg, fft_end, self.img_size[1], self.img_size[0])  # In pixels
        elif self.displacement_algorithm == "phase-correlation":
            raise NotImplementedError
        else:
            raise TypeError

        # Convert from pixels to millimeters (or equivalent):
        deltax, deltay = _deltax * self.xres, _deltay * self.yres
        self.current_position = self.current_position[0] + deltax, self.current_position[1] + deltay
        return deltax * self.xres, deltay * self.yres

    def save_config(self, path: str, filename="visual-odometer-config"):
        config = {
            "Displacement Algorithm": self.displacement_algorithm,
            "Frequency Window": self.frequency_window,
            "Spatial Window": self.spatial_window,
            "Image Size": self.img_size,
        }
        with open(path + "/" + filename + ".json", 'w') as fp:
            json.dump(config, fp)
