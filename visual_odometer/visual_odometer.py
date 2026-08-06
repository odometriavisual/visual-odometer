import numpy as np
import threading

from visual_odometer.displacement_estimators import svd_method
from visual_odometer.displacement_estimators import phase_correlation_method
from visual_odometer.displacement_estimators import proj_svd_method
from visual_odometer.displacement_estimators import phase_amplified_correlation_method
from visual_odometer.displacement_estimators import pc_analyze_image
from visual_odometer.dsp import crop_two_imgs_with_displacement


class VisualOdometer:
    """
    The class implementing the visual odometer.

    The visual odometer is capable of woking in the "Single Shot" mode and in the "Sequential" mode
    In the "Single Shot" mode, the visual odometer outputs the displacement between a pair of images.
    In the "Sequential" mode, the visual odometer outputs a stream of N-1 displacements from a sequence of N images.
    """

    def __init__(self, img_shape: (int, int), **kwargs):
        """
        Instantiates a visual odometer

        :param img_shape: The shape of the image array as defined by the numpy.ndarray.shape
        :param xres: Ratio of mm/pixels in the x dimension
        :param yres: Ratio of mm/pixels in the y dimension
        :param displacement_estimation_method:  Which displacement estimation method to be applied. Available methods: "svd", "phase-correlation", "projection-svd", "phase-amplified-correlation".
        :param reprocess_displacement: Set to True to enable double processing, double processing increases accuracy at the cost of processing time.
        :param frequency_window_method: Which frequency window to be applied. Available methods: “Stone_et_al_2001”, “ideal-lowpass”, None
        :param frequency_window_params: Parameters related to the chosen window.
        :param spatial_window_method: Which spatial window to be applied. Available methods: "blackman-harris", "raised-cosine", None
        :param spatial_window_params: Parameters related to the chosen window.
        :param downsampling_method: Which downsample algorithm to be applied. Available methods: “NN”, “bilinear”, "bicubic", None
        :param downsampling_params: Parameters related to the specific downsample algorithm.
        """
        # Default configs:
        self.configs = {
            "Displacement Estimation": {
                "method": kwargs.get("displacement_estimation_method", "svd"),
                "reprocess_displacement": kwargs.get("reprocess_displacement", False),
                "skip_frames": kwargs.get("skip_frames", False),
                "params": {
                    "skip_frames_threshold": 5,
                    "reprocess_displacement_count": 1
                },

            },
            "Frequency Window": {
                "method": kwargs.get("frequency_window_method", "Stone_et_al_2001"),
                "params": kwargs.get("frequency_window_params", {
                    "factor": 0.6,
                })
            },
            "Spatial Window": {
                "method": kwargs.get("spatial_window_method", "raised_cosine"),
                "params": kwargs.get("spatial_window_params", {
                    "a0": 0.358,
                    "a1": 0.47,
                    "a2": 0.135,
                    "a3": 0.037,
                })
            },
            "Downsampling": {
                "method": kwargs.get("downsampling_method", ""),
                "params": kwargs.get("downsampling_params", {
                    "factor": 1,
                })
            },
        }

        self.img_size = img_shape
        self.xres, self.yres = kwargs.get("xres", 1.), kwargs.get("yres", 1.)  # Relationship between displacement in pixels and millimeters

        self.current_position = np.array([0, 0])  # In pixels
        self.number_of_displacements = 0

        self.imgs_lock = threading.Lock()
        self.imgs_processed = [None, None]
        self.imgs_original = [None, None]

        method = self.configs["Displacement Estimation"]["method"]

        match method:
            case "svd":
                self.analyze_method = pc_analyze_image
                self.compute_displacement_method = lambda fft_beg, fft_end: svd_method(fft_beg, fft_end, self.img_size[1], self.img_size[0], phase_windowing="central")

            case "phase-correlation":
                self.analyze_method = pc_analyze_image
                self.compute_displacement_method = phase_correlation_method

            case "projection-svd":
                self.analyze_method = pc_analyze_image
                self.compute_displacement_method = lambda fft_beg, fft_end: proj_svd_method(fft_beg, fft_end, self.img_size[1], self.img_size[0], dx_max=30, dy_max=30, phase_windowing="central")

            case "phase-amplified-correlation":
                self.analyze_method = pc_analyze_image
                self.compute_displacement_method = lambda fft_beg, fft_end: phase_amplified_correlation_method(fft_beg, fft_end, gain=3)

            case _:
                raise ValueError(f"Displacement estimation method {method} not valid.")

        # The first img in imgs_processed will always be the last successful image used on a displacement estimation.
        # The second img will be the most recent image

    def estimate_displacement_between(self, img_beg, img_end) -> (float, float):
        """
        Estimates the displacement between two images

        Intended for the "Single Shot" mode, for estimating displacements between sequences of images use `estimate_last_displacement()`.

        :param img_beg: Image at t = t_0
        :param img_end: Image at t = t₀ + Δt
        :return: x and y displacements in mm
        """
        fft_beg = self.analyze_method(img_beg, self.configs)
        fft_end = self.analyze_method(img_end, self.configs)
        return self._estimate_displacement(fft_beg, fft_end)

    def _estimate_displacement(self, fft_beg, fft_end) -> (float, float):
        _deltax, _deltay = self.compute_displacement_method(fft_beg, fft_end)

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
        img_spectrum = self.analyze_method(img, self.configs)

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

