import numpy as np
import json
from .displacement_estimators.svd import svd_method
from .preprocessing import image_preprocessing
from .displacementprocessor import DisplacementProcessor
import threading
from collections import deque
import warnings


class VisualOdometer:
    def __init__(self, img_size,
                 xres=1, yres=1,
                 displacement_algorithm="svd",
                 frequency_window="Stone_et_al_2001",
                 spatial_window="blackman-harris",
                 num_threads=4, img_buffersize=5):

        self.displacement_algorithm = displacement_algorithm
        self.frequency_window = frequency_window
        self.spatial_window = spatial_window
        self.img_size = img_size
        self.xres, self.yres = xres, yres  # Relationship between displacement in pixels and millimeters

        self.current_position = (0, 0)  # In pixels
        self.imgs_processed = [None, None]
        # The first imgs_processed will always be the last successful image used on a displacement estimation. The second img will be the most recent image

        #
        self.displacements = deque()
        self.num_threads = num_threads  # How many threads are dedicated towards displacement estimation

    def __enter__(self):
        self._start_pool()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self._reset_pool()

    def calibrate(self, new_xres: float, new_yres: float):
        self.xres, self.yres = new_xres, new_yres

    def save_config(self, path: str, filename="visual-odometer-config"):
        config = {
            "Displacement Algorithm": self.displacement_algorithm,
            "Frequency Window": self.frequency_window,
            "Spatial Window": self.spatial_window,
            "Image Size": self.img_size,
        }
        with open(path + "/" + filename + ".json", 'w') as fp:
            json.dump(config, fp)

    def estimate_displacement_between(self, img_beg: np.ndarray, img_end: np.ndarray):
        """
        Estimates the displacement between img_beg and img_end.

        Intendend for single shot usage, for estimating displacements between sequences of images use estimate_last_displacement().
        """
        fft_beg = image_preprocessing(
            img_beg)  # dessa forma é sempre feito o preprocessamento EM DOBRO! (Duas vezes na mesma imagem)
        fft_end = image_preprocessing(img_end)  # dessa forma é sempre feito o preprocessamento EM DOBRO!

        return self._estimate_displacement(fft_beg, fft_end)

    def _estimate_displacement(self, fft_beg, fft_end) -> (float, float):
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

    def get_displacement(self):
        if len(self.displacements) > 0:
            return self.displacements.popleft()
        else:
            return None

    def feed_image(self, img: np.ndarray) -> None:
        # Update the latest image:
        if self.imgs_processed[0] is None:
            # The first iteration
            self.imgs_processed[0] = image_preprocessing(img)
        else:
            # Update the current image:
            self.imgs_processed[1] = image_preprocessing(img)

            # Assign an image pair to a worker:
            self._assign_worker()

    def _assign_worker(self):
        with self.lock:
            if self.pool:
                # Grab the pair of frames:
                img_beg = self.imgs_processed[0]
                img_end = self.imgs_processed[1]

                # Update the oldest image:
                self.imgs_processed[0] = img_end

                print("There is an available thread")
                self.processor = self.pool.pop()
                self.processor.feed_image(img_beg, img_end)
                self.processor.start_processing()
            else:
                warnings.warn("No available thread for displacement estimation.")
                self.processor = None

    def _start_pool(self):
        self.lock = threading.Lock()  # Lock for managing thread pool
        self.pool = [DisplacementProcessor(self) for i in range(self.num_threads)]
        self.processor = None
        self.terminated = False

    def _reset_pool(self):
        self.terminated = True
        # Guarantee that all workers returns to the pool:
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None

        # Now, empty the pool, joining each thread as we go
        while True:
            with self.lock:
                try:
                    processor = self.pool.pop()
                except IndexError:
                    break  # pool is empty
            processor.terminated = True
            processor.join()
