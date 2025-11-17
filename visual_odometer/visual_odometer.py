import numpy as np
import threading
import json
from abc import ABC, abstractmethod
from queue import Empty
from multiprocessing import Process, Queue as MPQueue, Pipe

from .preprocessing import image_preprocessing
from .dsp import crop_two_imgs_with_displacement
from .displacement_estimators import (
    svd_method,
    phase_correlation_method,
    proj_svd_method,
    phase_amplified_correlation_method
)


class DisplacementProcessor(ABC):
    """
    Abstract interface for displacement processors.
    Defines the methods that must be implemented in both synchronous and asynchronous modes, ensuring transparent switching between modes.
    """

    @abstractmethod
    def feed_image(self, img):
        """
        Sends an image for processing.

        :param img: Image to be processed (numpy array)
        """
        pass

    @abstractmethod
    def get_displacement(self):
        """
        Obtains the computed displacement (dx, dy) in mm.

        :return: Tuple (dx, dy) containing displacements in mm
        """
        pass

    @abstractmethod
    def shutdown(self):
        """Cleanly shuts down the processor."""
        pass


class SyncDisplacementProcessor(DisplacementProcessor):
    """
    Synchronous processor — performs all operations on the main thread in a blocking manner.

    In synchronous mode:
    - feed_image(): stores the preprocessed spectrum
    - get_displacement(): computes displacement between the two most recent images
    """
    def __init__(self, odometer):
        """
        Initializes the synchronous processor.

        :param odometer: Parent VisualOdometer instance
        """
        self.odometer = odometer
        self.imgs_processed = [None, None]  # Espectros preprocessados
        self.imgs_original = [None, None]  # Imagens originais (para reprocessamento)
        self.imgs_lock = threading.Lock()

    def feed_image(self, img):
        """
        Stores the preprocessed image for later displacement calculation.
        The first image serves as reference. Subsequent images are compared to the reference when get_displacement() is called.

        :param img: Next image in the sequence (numpy array)
        """
        # Preprocesses the image (applies windowing, downsampling, FFT, etc.)
        img_spectrum = image_preprocessing(img, self.odometer.configs)

        if self.imgs_processed[0] is None:
            # First iteration — stores as reference
            self.imgs_processed[0] = img_spectrum
            self.imgs_original[0] = img
        else:
            # Updates the most recent image
            with self.imgs_lock:
                self.imgs_processed[1] = img_spectrum
                self.imgs_original[1] = img

    def get_displacement(self):
        """
        Computes the displacement between the reference image and the most recent one.

        This method:
        1. Estimates the raw displacement between spectra
        2. Optionally reprocesses using cropped images for higher accuracy
        3. Optionally discards frames with displacement below a threshold
        4. Updates the accumulated odometer position

        :return: Tuple (dx, dy) with displacement in mm
        """
        try:
            reprocess_displacement = self.odometer.configs["Displacement Estimation"]["reprocess_displacement"]
            skip_frames = self.odometer.configs["Displacement Estimation"]["skip_frames"]

            # Checks whether two images are available for comparison
            if self.imgs_processed[0] is None or self.imgs_processed[1] is None:
                return 0.0, 0.0

            spectrum_beg = self.imgs_processed[0]
            original_img_beg = self.imgs_original[0]

            with self.imgs_lock:
                spectrum_end = self.imgs_processed[1].copy()
                original_img_end = self.imgs_original[1].copy()

            # 1. Estimate raw displacement between spectra
            displacement = self.odometer._estimate_displacement(spectrum_beg, spectrum_end)

            # 2. Optional reprocessing for higher accuracy
            # Crops the original images and recalculates displacement
            if reprocess_displacement:
                count = self.odometer.configs["Displacement Estimation"]["params"].get("reprocess_displacement_count",
                                                                                       1)
                for _ in range(count):
                    round_dx = int(round(displacement[0] / self.odometer.xres))
                    round_dy = int(round(displacement[1] / self.odometer.yres))

                    crop_img_beg, crop_img_end = crop_two_imgs_with_displacement(
                        original_img_beg, original_img_end, round_dx, round_dy
                    )

                    new_displacement = self.odometer.estimate_displacement_between(crop_img_beg, crop_img_end)
                    displacement = [round_dx * self.odometer.xres + new_displacement[0],
                                    round_dy * self.odometer.yres + new_displacement[1]]

            # 3. Skip frames: discards frames with very small displacement
            if skip_frames:
                threshold = self.odometer.configs["Displacement Estimation"]["params"]["skip_frames_threshold"]
                if np.linalg.norm(displacement) < threshold:
                    # Não atualiza a imagem base (mantém img_beg como referência)
                    return 0.0, 0.0

            # 4. Updates the base image only if the displacement was accepted
            self.imgs_processed[0] = spectrum_end
            self.imgs_original[0] = original_img_end

            # 5. Updates the odometer's accumulated position
            self.odometer.current_position[0] += displacement[0]
            self.odometer.current_position[1] += displacement[1]
            self.odometer.number_of_displacements += 1

            return tuple(displacement)

        except NotImplementedError:
            return None, None

    def shutdown(self):
        """"Nothing to do in synchronous mode — no processes or threads to shut down."""
        pass


# ============================================================================
# IMPLEMENTAÇÃO ASSÍNCRONA (NÃO-BLOQUEANTE)
# ============================================================================

def _async_preprocessing_worker(conn_in, conn_out, configs):
    """
    Worker that performs only the image preprocessing.

    This worker runs in a separate process and is responsible for:
    - Receiving images from the input pipe
    - Applying preprocessing (windowing, FFT, etc.)
    - Sending the spectrum AND the original image to the next worker

    :param conn_in: Input pipe connection receiving raw images
    :param conn_out: Output pipe connection sending (spectrum, img_original)
    :param configs: Odometer configuration dictionary
    """
    while True:
        try:
            img = conn_in.recv()

            if img is None:  # Shutdown signal
                conn_out.send(None)
                break

            # Preprocess the image
            spectrum = image_preprocessing(img, configs)

            # Send the spectrum and the original image
            conn_out.send((spectrum, img))

        except EOFError:
            break
        except Exception as e:
            print(f"Error in preprocessing worker: {e}")
            break


def _async_displacement_worker(conn_in, conn_out, configs, xres, yres):
    """
    Worker that computes displacement between pairs of images.

    This worker runs in a separate process and is responsible for:
    - Receiving preprocessed spectra from the input pipe
    - Computing displacement between consecutive image pairs
    - Optionally reprocessing with cropping for higher accuracy
    - Sending the computed displacement to the output pipe

    :param conn_in: Input pipe connection receiving (spectrum, img_original)
    :param conn_out: Output pipe connection sending (dx_mm, dy_mm)
    :param configs: Odometer configuration dictionary
    :param xres: Resolution in X (mm/pixel)
    :param yres: Resolution in Y (mm/pixel)
    """
    prev_spectrum = None
    prev_img = None

    while True:
        try:
            data = conn_in.recv()

            if data is None:  # Shutdown signal
                conn_out.send(None)
                break

            spectrum, img = data

            # Initialize dx_mm and dy_mm as 0.0 by default
            dx_mm, dy_mm = 0.0, 0.0

            if prev_spectrum is not None:
                # Calculate displacement between the previous and current images
                method = configs["Displacement Estimation"]["method"]
                img_size_x = img.shape[1]
                img_size_y = img.shape[0]

                # 1. Initial displacement estimate
                if method == "svd":
                    dx, dy = svd_method(prev_spectrum, spectrum, img_size_x, img_size_y, phase_windowing="central")
                elif method == "phase-correlation":
                    dx, dy = phase_correlation_method(prev_spectrum, spectrum)
                elif method == "projection-svd":
                    dx, dy = proj_svd_method(
                        prev_spectrum, spectrum, img_size_x, img_size_y,
                        dx_max=60, dy_max=60, phase_windowing="central"
                    )
                elif method == "phase-amplified-correlation":
                    dx, dy = phase_amplified_correlation_method(prev_spectrum, spectrum, gain=3)
                else:
                    dx, dy = 0, 0

                # 2. Optional reprocessing for greater accuracy
                if configs["Displacement Estimation"].get("reprocess_displacement", False):
                    count = configs["Displacement Estimation"]["params"].get("reprocess_displacement_count", 1)

                    for _ in range(count):
                        round_dx = int(round(dx))
                        round_dy = int(round(dy))

                        # Crop the original images
                        crop_img_beg, crop_img_end = crop_two_imgs_with_displacement(
                            prev_img, img, round_dx, round_dy
                        )

                        # Recalculate the spectrum for the crops
                        spectrum_beg_crop = image_preprocessing(crop_img_beg, configs)
                        spectrum_end_crop = image_preprocessing(crop_img_end, configs)

                        # Recalculate the displacement with greater precision
                        if method == "svd":
                            dx_ref, dy_ref = svd_method(
                                spectrum_beg_crop, spectrum_end_crop,
                                crop_img_end.shape[1], crop_img_end.shape[0],
                                phase_windowing="central"
                            )
                        elif method == "phase-correlation":
                            dx_ref, dy_ref = phase_correlation_method(spectrum_beg_crop, spectrum_end_crop)
                        elif method == "projection-svd":
                            dx_ref, dy_ref = proj_svd_method(
                                spectrum_beg_crop, spectrum_end_crop,
                                crop_img_end.shape[1], crop_img_end.shape[0],
                                dx_max=60, dy_max=60, phase_windowing="central"
                            )
                        elif method == "phase-amplified-correlation":
                            dx_ref, dy_ref = phase_amplified_correlation_method(spectrum_beg_crop, spectrum_end_crop, gain=3)
                        else:
                            dx_ref, dy_ref = 0, 0

                        dx = round_dx + dx_ref
                        dy = round_dy + dy_ref

                # 3. Convert from pixels to mm
                dx_mm = dx * xres
                dy_mm = dy * yres

                # 4. Skip frames: discard very small displacements
                if configs["Displacement Estimation"].get("skip_frames", False):
                    threshold = configs["Displacement Estimation"]["params"].get("skip_frames_threshold", 5)
                    if np.sqrt(dx_mm ** 2 + dy_mm ** 2) < threshold:
                        # Do not update prev_spectrum (keeps the previous reference)
                        continue

                # 5. Send the calculated displacement
                conn_out.send((dx_mm, dy_mm))

            # 6. Update the references for the next iteration
            prev_spectrum = spectrum
            prev_img = img

        except EOFError:
            break
        except Exception as e:
            print(f"Error in displacement worker: {e}")
            break

class AsyncDisplacementProcessor(DisplacementProcessor):
    """
    Asynchronous processor — processes images in separate processes using Pipes.

    In asynchronous mode:
    - feed_image(): sends images to the preprocessing pipe (non-blocking)
    - get_displacement(): returns all displacements accumulated since the last call

    Architecture:
    1. Main thread → feed_image() → Pipe 1 → Worker 1
    2. Worker 1 (process) → preprocessing → Pipe 2 → Worker 2
    3. Worker 2 (process) → displacement calculation → Pipe 3 → Main thread
    4. Consumer thread → accumulates displacement
    5. Main thread → get_displacement() → returns accumulated displacement

    This mode allows the camera to capture images rapidly while heavy processing
    happens in parallel using pipes for inter-process communication.
    """

    def __init__(self, odometer):
        """
        Initializes the asynchronous processor with workers running in separate processes.

        :param odometer: Parent VisualOdometer instance
        """
        self.odometer = odometer

        # Thread-safe displacement accumulator
        self.accumulated_displacement = [0.0, 0.0]
        self.displacement_lock = threading.Lock()

        # Create pipes for inter-process communication
        self.pipe_main_to_pre_send, pipe_main_to_pre_recv = Pipe()
        pipe_pre_to_disp_send, pipe_pre_to_disp_recv = Pipe()
        pipe_disp_to_main_send, self.pipe_disp_to_main_recv = Pipe()

        # Create worker processes
        self.preprocessing_process = Process(
            target=_async_preprocessing_worker,
            args=(pipe_main_to_pre_recv, pipe_pre_to_disp_send, odometer.configs),
            daemon=True,
        )
        self.displacement_process = Process(
            target=_async_displacement_worker,
            args=(
                pipe_pre_to_disp_recv,
                pipe_disp_to_main_send,
                odometer.configs,
                odometer.xres,
                odometer.yres
            ),
            daemon=True,
        )

        # Start worker processes
        self.preprocessing_process.start()
        self.displacement_process.start()

        # Thread responsible for consuming results from the workers
        self.result_thread = threading.Thread(target=self._consume_results, daemon=True)
        self.result_thread.start()

    def feed_image(self, img):
        """
        Sends an image for asynchronous (non-blocking) processing.

        The image is sent to the preprocessing pipe. If sending fails,
        the image is silently discarded, ensuring the acquisition is not blocked.

        :param img: Next image in the sequence (numpy array)
        """
        try:
            # Ensure a contiguous array for better pickle serialization
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)

            # Send image through pipe
            self.pipe_main_to_pre_send.send(img)
        except Exception:
            # Silently ignore errors (e.g., broken pipe, dead worker process)
            pass

    def get_displacement(self):
        """
        Retrieves the accumulated displacement since the last call.

        This method:
        1. Returns accumulated displacement from all processed image pairs
        2. Resets the accumulator to zero
        3. Updates the odometer's position and counter

        :return: Tuple (dx_mm, dy_mm) with displacement in mm
        """
        with self.displacement_lock:
            dx, dy = self.accumulated_displacement
            self.accumulated_displacement = [0.0, 0.0]

        # Update odometer position and counter if there was displacement
        if dx != 0.0 or dy != 0.0:
            self.odometer.current_position[0] += dx
            self.odometer.current_position[1] += dy
            self.odometer.number_of_displacements += 1

        return dx, dy

    def _consume_results(self):
        """
        Thread that consumes results from the workers and accumulates displacements.

        This thread runs continuously in the background, collecting displacement values
        from the output pipe and accumulating them in a thread-safe manner.
        """
        while True:
            try:
                displacement = self.pipe_disp_to_main_recv.recv()

                if displacement is None:  # Shutdown signal
                    break

                dx, dy = displacement

                # Accumulate the displacement in a thread-safe manner
                with self.displacement_lock:
                    self.accumulated_displacement[0] += dx
                    self.accumulated_displacement[1] += dy

            except EOFError:
                # Pipe was closed
                break
            except Exception as e:
                print(f"Error consuming results: {e}")
                break

    def shutdown(self):
        """
        Shuts down workers and threads cleanly.

        Procedure:
        1. Sends a stop signal (None) to the workers
        2. Waits for graceful termination of the processes
        3. Forces termination if necessary (terminate)
        4. Closes communication pipes
        """
        try:
            # Send stop signal to the first worker
            self.pipe_main_to_pre_send.send(None)

            # Wait for process completion
            self.preprocessing_process.join(timeout=2)
            self.displacement_process.join(timeout=2)

            # Force termination if they are still alive
            if self.preprocessing_process.is_alive():
                self.preprocessing_process.terminate()
                self.preprocessing_process.join(timeout=1)

            if self.displacement_process.is_alive():
                self.displacement_process.terminate()
                self.displacement_process.join(timeout=1)

            # Close pipes
            self.pipe_main_to_pre_send.close()
            self.pipe_disp_to_main_recv.close()

        except Exception as e:
            print(f"Error shutting down async processor: {e}")

class VisualOdometer:
    """
    The class implementing the visual odometer.

    The visual odometer is capable of working in the "Single Shot" mode and in the "Sequential" mode
    In the "Single Shot" mode, the visual odometer outputs the displacement between a pair of images.
    In the "Sequential" mode, the visual odometer outputs a stream of N-1 displacements from a sequence of N images.

    Modos de operação:
    - Síncrono (async_mode=False): Processamento bloqueante no thread principal
    - Assíncrono (async_mode=True): Processamento em paralelo com workers separados
    """

    def __init__(self, img_shape: tuple, xres: float = 1.0, yres: float = 1.0,
                 configs: dict = None, async_mode=False):
        """
        Instantiates a visual odometer

        :param img_shape: The shape of the image array as defined by the numpy.ndarray.shape
        :param xres: Ratio of mm/pixels in the x dimension
        :param yres: Ratio of mm/pixels in the y dimension
        :param configs: Dicionário de configurações (se None, usa configurações padrão)
        :param async_mode: Se True, usa processamento assíncrono; se False, usa síncrono
        """
        from .utils import DEFAULT_CONFIG, merge_dicts

        # Default configuration
        self.configs = DEFAULT_CONFIG.copy()
        if configs:
            merge_dicts(self.configs, configs)

        # Basic parameters
        self.img_size = img_shape
        self.xres, self.yres = xres, yres  # Relationship between pixel displacement and millimeters

        # Odometer state (accumulated position)
        self.current_position = np.array([0.0, 0.0])  # In mm
        self.number_of_displacements = 0

        # Chooses processor based on mode
        if async_mode:
            self.processor = AsyncDisplacementProcessor(self)
        else:
            self.processor = SyncDisplacementProcessor(self)

    def feed_image(self, img):
        """
        Send the next image in "Sequential" mode for processing.

        Behavior depends on the mode:
        - Synchronous: Stores the preprocessed spectrum
        - Asynchronous: Sends the image to the preprocessing queue (non-blocking)

        :param img: Next image in the stream (numpy array)
        """
        self.processor.feed_image(img)

    def get_displacement(self):
        """
        Get the next displacement in "Sequential" mode.

        Behavior depends on the mode:
        - Synchronous: Computes displacement between the two most recent images
        - Asynchronous: Returns the displacement accumulated since the last call

        :return: Next x and y displacements in mm (tuple (dx, dy))
        """
        return self.processor.get_displacement()

    def get_position(self):
        """
        Returns the current accumulated odometer position.

        :return: NumPy array with [x, y] in mm
        """
        return self.current_position.copy()

    def estimate_displacement_between(self, img_beg, img_end):
        """
        Estimates the displacement between two images.

        Intended for the "Single Shot" mode — for estimating displacement in continuous
        image sequences, use `get_displacement()` instead.

        This method does NOT affect the sequential odometer state (current_position).
        It is useful for isolated analyses or arbitrary pairwise image comparisons.

        :param img_beg: Image at t = t₀
        :param img_end: Image at t = t₀ + Δt
        :return: x and y displacements in mm (tuple (dx, dy))
        """
        img_x_size = img_beg.shape[1]
        img_y_size = img_beg.shape[0]

        fft_beg = image_preprocessing(img_beg, self.configs)
        fft_end = image_preprocessing(img_end, self.configs)

        return self._estimate_displacement(fft_beg, fft_end, img_x_size, img_y_size)

    def _estimate_displacement(self, fft_beg, fft_end, img_size_x=None, img_size_y=None):
        """
        Internal calculation of displacement between two spectra.

        This method:
        1. Selects the configured estimation method
        2. Computes the displacement in pixels
        3. Converts the result to millimeters using xres and yres

        Note: This method does NOT update current_position. It is the caller’s
        responsibility to update the odometer position if needed.

        :param fft_beg: Spectrum of the initial image
        :param fft_end: Spectrum of the final image
        :param img_size_x: Image width in pixels
        :param img_size_y: Image height in pixels
        :return: Tuple (dx_mm, dy_mm) containing displacement in millimeters
        """
        method = self.configs["Displacement Estimation"]["method"]

        if img_size_x is None:
            img_size_x = self.img_size[1]
            img_size_y = self.img_size[0]

        # Computes pixel displacement using the configured method
        match method:
            case "svd":
                _deltax, _deltay = svd_method(fft_beg, fft_end, img_size_x, img_size_y, phase_windowing="central")
            case "phase-correlation":
                _deltax, _deltay = phase_correlation_method(fft_beg, fft_end)
            case "projection-svd":
                _deltax, _deltay = proj_svd_method(fft_beg, fft_end, img_size_x, img_size_y, dx_max=30, dy_max=30,
                                                   phase_windowing="central")
            case "phase-amplified-correlation":
                _deltax, _deltay = phase_amplified_correlation_method(fft_beg, fft_end, gain=3)
            case _:
                raise ValueError(f"Displacement estimation method {method} not valid.")

        # Converts from pixels to millimeters (or equivalent unit)
        deltax, deltay = _deltax * self.xres, _deltay * self.yres

        return deltax, deltay

    def calibrate(self, new_xres: float, new_yres: float):
        """
        Changes the spatial resolution of the odometer.

        :param new_xres: New resolution in X (mm/pixel)
        :param new_yres: New resolution in Y (mm/pixel)
        """
        self.xres, self.yres = new_xres, new_yres

    def config_displacement_estimation(self, method: str = "", **kwargs):
        """
        Configures the displacement estimation method.

        :param method: Method to be used ("svd", "phase-correlation", "projection-svd", "phase-amplified-correlation")
        :param kwargs: Additional parameters for the method
        """
        if method:
            self.configs["Displacement Estimation"]["method"] = method
        if kwargs:
            self.configs["Displacement Estimation"]["params"].update(kwargs)

    def config_frequency_window(self, method: str = "", **kwargs):
        """
        Configures the frequency window applied to spectra.

        :param method: Windowing method ("Stone_et_al_2001", "ideal-lowpass", None)
        :param kwargs: Window parameters
        """
        if method:
            self.configs["Frequency Window"]["method"] = method
        if kwargs:
            self.configs["Frequency Window"]["params"].update(kwargs)

    def config_spatial_window(self, method: str = "", **kwargs):
        """
        Configures the spatial window applied to images.

        :param method: Windowing method ("blackman-harris", "raised-cosine", None)
        :param kwargs: Window parameters
        """
        if method:
            self.configs["Spatial Window"]["method"] = method
        if kwargs:
            self.configs["Spatial Window"]["params"].update(kwargs)

    def config_downsampling(self, method: str = "", **kwargs):
        """
        Configures the image downsampling method.

        :param method: Downsampling method ("NN", "bilinear", "bicubic", None)
        :param kwargs: Downsampling parameters (e.g., factor)
        """
        if method:
            self.configs["Downsampling"]["method"] = method
        if kwargs:
            self.configs["Downsampling"]["params"].update(kwargs)

    def set_config(self, new_config: dict):
        """
        Overrides all odometer configurations.

        :param new_config: Dictionary containing the new configuration values
        """
        from .utils import merge_dicts
        merge_dicts(self.configs, new_config)

    def print_config(self):
        """Prints the current configuration in JSON format."""
        print(json.dumps(self.configs, indent=2))


    def save_config(self, path: str, filename="visual-odometer-config"):
        """
        Saves the current configuration to a JSON file.

        :param path: Directory path where the file should be saved
        :param filename: File name (without extension)
        :return: True if saved successfully, otherwise False
        """
        try:
            with open(path + "/" + filename + ".json", 'w') as fp:
                json.dump(self.configs, fp, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def shutdown(self):
        """
        Shuts down the odometer cleanly.

        In synchronous mode: Does nothing (no resources to release)
        In asynchronous mode: Shuts down worker processes and threads
        """
        self.processor.shutdown()

    def __del__(self):
        """
        Destructor to ensure clean shutdown when the object is deleted.
        """

        self.shutdown()