from multiprocessing import Pipe, Process
import numpy as np
import json
import threading

from .displacement_estimators import svd_method
from .displacement_estimators import phase_correlation_method
from .preprocessing import image_preprocessing
from .dsp import crop_two_imgs_with_displacement

DEFAULT_CONFIG = {
    "Displacement Estimation": {
        "method": "svd",
        "reprocess_displacement": True,
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


# fora da classe
def worker_img_preprocess(conn_in, conn_out, configs):
    while True:
        img = conn_in.recv()
        if img is None:
            conn_out.send(None)
            break
        spectrum = image_preprocessing(img, configs)
        conn_out.send((spectrum, img))


def worker_svd(conn_in, conn_out, configs, xres, yres):
    prev_spectrum, prev_img = None, None
    acc_disp = [0.0, 0.0]

    while True:
        data = conn_in.recv()
        if data is None:
            conn_out.send(None)
            break

        spectrum, img = data
        if prev_spectrum is not None:
            method = configs["Displacement Estimation"]["method"]
            if method == "svd":
                dx, dy = svd_method(prev_spectrum, spectrum, img.shape[1], img.shape[0])
            elif method == "phase-correlation":
                dx, dy = phase_correlation_method(prev_spectrum, spectrum)
            else:
                raise NotImplementedError

            dx *= xres
            dy *= yres

            acc_disp = [acc_disp[0] + dx, acc_disp[1] + dy]

        prev_spectrum, prev_img = spectrum, img
        conn_out.send(tuple(acc_disp))
        acc_disp = [0.0, 0.0]


class VisualOdometer:

    def __init__(self, img_shape: (int, int), xres: float = 1.0, yres: float = 1.0, async_mode=False):
        """
        Instantiates a visual odometer
        :param img_shape: The shape of the image array as defined by the numpy.ndarray.shape
        :param xres: Ratio of mm/pixels in the x dimension
        :param yres: Ratio of mm/pixels in the y dimension
        :param async_mode: Se True, usa processamento assíncrono com pipes
        """
        # Default configs:
        self.configs = DEFAULT_CONFIG

        self.img_size = img_shape
        self.xres, self.yres = xres, yres  # Relationship between displacement in pixels and millimeters

        self.current_position = np.array([0.0, 0.0])  # In pixels
        self.number_of_displacements = 0

        self.imgs_lock = threading.Lock()
        self.imgs_processed = [None, None]
        self.imgs_original = [None, None]

        self.async_mode = async_mode

        if async_mode:
            self._setup_async_mode()
        else:
            self._setup_sync_mode()

    def _setup_async_mode(self):
        """Configura o modo assíncrono com pipes e workers."""
        self.accumulated_displacements = [0.0, 0.0]
        self.displacement_lock = threading.Lock()  # ADICIONADO: Lock para deslocamentos acumulados

        # Criação dos pipes
        # Pipe 1: main -> preprocess worker
        self.pipe_main_to_pre_send, pipe_main_to_pre_recv = Pipe()

        # Pipe 2: preprocess worker -> svd worker
        pipe_pre_to_svd_send, pipe_pre_to_svd_recv = Pipe()

        # Pipe 3: svd worker -> main
        pipe_svd_to_main_send, self.pipe_svd_to_main_recv = Pipe()

        # Criação dos processos workers
        self.proc_preprocess = Process(
            target=worker_img_preprocess,
            args=(pipe_main_to_pre_recv, pipe_pre_to_svd_send, self.configs),
            daemon=True,
        )

        self.proc_svd = Process(
            target=worker_svd,
            args=(pipe_pre_to_svd_recv, pipe_svd_to_main_send, self.configs, self.xres, self.yres),
            daemon=True,
        )

        # Inicia os processos
        self.proc_preprocess.start()
        self.proc_svd.start()

        # ADICIONADO: Thread para receber deslocamentos dos workers
        self.result_thread = threading.Thread(target=self._displacement_receiver, daemon=True)
        self.result_thread.start()

        # Bind dos métodos assíncronos
        self.feed_image = self._feed_image_async
        self.get_displacement = self._get_displacement_async

        print("Modo assíncrono inicializado com sucesso!")

    def _displacement_receiver(self):
        """
        Thread que recebe deslocamentos dos workers e os acumula.
        """
        while True:
            try:
                displacement = self.pipe_svd_to_main_recv.recv()
                if displacement is None:
                    break

                with self.displacement_lock:
                    self.accumulated_displacements[0] += displacement[0]
                    self.accumulated_displacements[1] += displacement[1]
            except Exception as e:
                print(f"Erro ao receber deslocamento: {e}")
                break

    def _setup_sync_mode(self):
        """Configura o modo síncrono."""
        # Bind dos métodos síncronos
        self.feed_image = self._feed_image_sync
        self.get_displacement = self._get_displacement_sync

        print("Modo síncrono inicializado.")

    def _feed_image_async(self, img):
        """
        Envia imagem para processamento assíncrono.
        :param img: Imagem a ser processada
        """
        try:
            self.pipe_main_to_pre_send.send(img)
        except Exception as e:
            print(f"Erro ao enviar imagem: {e}")

    def _get_displacement_async(self):
        """
        Obtém e zera os deslocamentos acumulados.
        :return: Tupla (dx, dy) com deslocamentos acumulados em mm
        """
        try:
            with self.displacement_lock:
                displacement = tuple(self.accumulated_displacements)
                self.accumulated_displacements = [0.0, 0.0]  # Zera após obter

            self.current_position += displacement
            self.number_of_displacements += 1
            return displacement

        except Exception as e:
            print(f"Erro ao obter deslocamento: {e}")
            return 0.0, 0.0

    def _feed_image_sync(self, img) -> None:
        """
        Send the next image in "Sequential" mode for processing
        :param img: Next image in the stream
        """
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

    def _get_displacement_sync(self):
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
                        crop_img_beg, crop_img_end = crop_two_imgs_with_displacement(
                            original_img_beg, original_img_end, round_dx, round_dy
                        )
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
        img_y_size = img_beg.shape[0]  # Corrigido: era img_end.shape[0]

        fft_beg = image_preprocessing(img_beg, self.configs)
        fft_end = image_preprocessing(img_end, self.configs)
        return self._estimate_displacement(fft_beg, fft_end, img_x_size, img_y_size)

    def _estimate_displacement(self, fft_beg, fft_end, img_size_x=None, img_size_y=None) -> (float, float):
        method = self.configs["Displacement Estimation"]["method"]

        if img_size_x is None:
            img_size_x = self.img_size[1]
            img_size_y = self.img_size[0]

        if method == "svd":
            _deltax, _deltay = svd_method(fft_beg, fft_end, img_size_x, img_size_y)  # In pixels
        elif method == "phase-correlation":
            _deltax, _deltay = phase_correlation_method(fft_beg, fft_end)
        else:
            raise NotImplementedError

        # Convert from pixels to millimeters (or equivalent):
        deltax, deltay = _deltax * self.xres, _deltay * self.yres
        return deltax, deltay

    def shutdown(self):
        """
        Encerra os processos workers de forma limpa (apenas no modo assíncrono).
        """
        if self.async_mode:
            try:
                # Envia sinal de parada para os workers
                self.pipe_main_to_pre_send.send(None)

                # Aguarda os processos terminarem
                self.proc_preprocess.join(timeout=5)
                self.proc_svd.join(timeout=5)

                # Força término se necessário
                if self.proc_preprocess.is_alive():
                    self.proc_preprocess.terminate()
                if self.proc_svd.is_alive():
                    self.proc_svd.terminate()

                print("Workers encerrados com sucesso!")
            except Exception as e:
                print(f"Erro ao encerrar workers: {e}")

    def __del__(self):
        """Destructor - garante que os processos sejam encerrados."""
        if hasattr(self, 'async_mode') and self.async_mode:
            self.shutdown()

    # Métodos de configuração
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
        print(json.dumps(self.configs, indent=2))

    def save_config(self, path: str, filename="visual-odometer-config"):
        try:
            with open(path + "/" + filename + ".json", 'w') as fp:
                json.dump(self.configs, fp, indent=2)
            return True
        except Exception as e:
            print(f"Erro ao salvar config: {e}")
            return False