# visual_odometer/core.py

import numpy as np
import threading
import json

from .utils import merge_dicts  # Importa de utils
# Importa os estimadores (SVF, PC, etc.) para os métodos de shot único
from .displacement_estimators import svd_method, phase_correlation_method, proj_svd_method, \
    phase_amplified_correlation_method
from .preprocessing import image_preprocessing

# Classe que será usada para ligar os hooks (síncrono ou assíncrono)
AsyncOdometerHooks = None
SyncOdometerHooks = None


class VisualOdometer:
    """
    A classe implementando o visual odometer.
    """

    def __init__(self, img_shape: (int, int), xres: float = 1.0, yres: float = 1.0, configs: dict = None,
                 async_mode=False):

        from .utils import DEFAULT_CONFIG
        # Importa os hooks após carregar as classes de worker para evitar erro de importação circular
        global AsyncOdometerHooks, SyncOdometerHooks
        from .async_mode import AsyncOdometerHooks
        from .sync_mode import SyncOdometerHooks

        self.configs = DEFAULT_CONFIG.copy()
        if configs:
            merge_dicts(self.configs, configs)

        self.img_size = img_shape
        self.xres, self.yres = xres, yres

        # Estado Sequencial (compartilhado, mas gerenciado pelos hooks)
        self.current_position = np.array([0.0, 0.0])
        self.number_of_displacements = 0

        self.imgs_lock = threading.Lock()
        self.imgs_processed = [None, None]
        self.imgs_original = [None, None]

        self.async_mode = async_mode
        self.hooks = None

        if async_mode:
            self.hooks = AsyncOdometerHooks(self)
        else:
            self.hooks = SyncOdometerHooks(self)

    # --- Métodos Independentes de Modo ---

    def _estimate_displacement(self, fft_beg, fft_end, img_size_x=None, img_size_y=None) -> (float, float):
        """Cálculo interno de deslocamento, usado por estimate_displacement_between."""
        method = self.configs["Displacement Estimation"]["method"]

        if img_size_x is None:
            img_size_x = self.img_size[1]
            img_size_y = self.img_size[0]

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

        # Converte de pixels para mm/unidade
        deltax, deltay = _deltax * self.xres, _deltay * self.yres
        return deltax, deltay

    def estimate_displacement_between(self, img_beg, img_end) -> (float, float):
        """
        Modo "Single Shot": Estima o deslocamento entre duas imagens quaisquer.
        """
        img_x_size = img_beg.shape[1]
        img_y_size = img_beg.shape[0]

        fft_beg = image_preprocessing(img_beg, self.configs)
        fft_end = image_preprocessing(img_end, self.configs)

        # Este método NÃO atualiza current_position
        return self._estimate_displacement(fft_beg, fft_end, img_x_size, img_y_size)

    def calibrate(self, new_xres: float, new_yres: float):
        """Altera a resolução (mm/pixel)."""
        self.xres, self.yres = new_xres, new_yres

    # --- Métodos de Configuração ---

    def _config(self, section: str, method: str = "", **kwargs):
        """Método auxiliar para configurar seções."""
        if method:
            self.configs[section]["method"] = method
        if kwargs:
            self.configs[section]["params"].update(kwargs)

    def config_displacement_estimation(self, method: str = "", **kwargs):
        self._config("Displacement Estimation", method, **kwargs)

    def config_frequency_window(self, method: str = "", **kwargs):
        self._config("Frequency Window", method, **kwargs)

    def config_spatial_window(self, method: str = "", **kwargs):
        self._config("Spatial Window", method, **kwargs)

    def config_downsampling(self, method: str = "", **kwargs):
        self._config("Downsampling", method, **kwargs)

    def set_config(self, new_config: dict):
        """Sobrescreve todas as configurações."""
        from .utils import merge_dicts
        merge_dicts(self.configs, new_config)

    def print_config(self):
        print(json.dumps(self.configs, indent=2))

    def save_config(self, path: str, filename="visual-odometer-config"):
        from .utils import save_config
        return save_config(self.configs, path, filename)

    def shutdown(self):
        """Encerra processos de forma limpa (se estiver no modo assíncrono)."""
        if self.hooks:
            # Chama o método shutdown específico do hook (async ou sync, mas o sync não faz nada)
            if hasattr(self.hooks, 'shutdown'):
                self.hooks.shutdown()

    def __del__(self):
        """Destructor para garantir o encerramento limpo."""
        self.shutdown()
