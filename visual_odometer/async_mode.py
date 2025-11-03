import threading
from multiprocessing import Pipe, Process
import numpy as np
from .displacement_estimators import svd_method, phase_correlation_method, proj_svd_method
from .preprocessing import image_preprocessing
from .dsp import crop_two_imgs_with_displacement

#Todo: Atualizar esse código para funcionar com os novos métodos de estimação

# --- Worker Functions ---
def worker_img_preprocess(conn_in, conn_out, configs):
    """Processa imagens recebidas do pipe principal e envia o espectro para o worker SVD."""
    while True:
        img = conn_in.recv()
        if img is None:
            conn_out.send(None)
            break
        spectrum = image_preprocessing(img, configs)
        conn_out.send((spectrum, img))  # Envia espectro E a imagem original (necessária para reprocessamento)


def worker_svd(conn_in, conn_out, configs, xres, yres):
    """Recebe espectros, calcula deslocamentos e envia o deslocamento líquido para o receiver."""
    prev_spectrum, prev_img = None, None
    # Acumulador de deslocamento *dentro do worker* (apenas para refinamento entre duas imagens)
    acc_disp_local = [0.0, 0.0]

    while True:
        data = conn_in.recv()
        if data is None:
            conn_out.send(None)
            break

        spectrum, img = data

        # Inicializa dx_mm e dy_mm como 0.0 por padrão
        dx_mm, dy_mm = 0.0, 0.0

        if prev_spectrum is not None:
            method = configs["Displacement Estimation"]["method"]
            img_size_x = img.shape[1]
            img_size_y = img.shape[0]

            if method == "svd":
                dx, dy = svd_method(prev_spectrum, spectrum, img_size_x, img_size_y)
            elif method == "phase-correlation":
                dx, dy = phase_correlation_method(prev_spectrum, spectrum)
            elif method == "projection-svd":
                dx, dy = proj_svd_method(prev_spectrum, spectrum, img_size_x, img_size_y, dx_max=30, dy_max=30,
                                                   phase_windowing="central") #Implementando temporariamente pra testar
            else:
                raise NotImplementedError(f"Método {method} não implementado no worker SVD.")

            # 2. Reprocessamento opcional
            if configs["Displacement Estimation"].get("reprocess_displacement", False):
                count = configs["Displacement Estimation"]["params"].get("reprocess_displacement_count", 1)

                for _ in range(count):
                    round_dx = int(round(dx))
                    round_dy = int(round(dy))

                    crop_img_beg, crop_img_end = crop_two_imgs_with_displacement(
                        prev_img, img, round_dx, round_dy
                    )

                    # Recalcula o espectro para os crops no loop de refinamento
                    dx_ref, dy_ref = svd_method(
                        image_preprocessing(crop_img_beg, configs),
                        image_preprocessing(crop_img_end, configs),
                        crop_img_end.shape[1],
                        crop_img_end.shape[0],
                    )

                    dx = round_dx + dx_ref
                    dy = round_dy + dy_ref

            # 3. Aplica resolução (pixels -> mm)
            dx_mm, dy_mm = dx * xres, dy * yres

            # 4. Acumula para envio ao thread principal
            acc_disp_local = [acc_disp_local[0] + dx_mm, acc_disp_local[1] + dy_mm]

        prev_spectrum, prev_img = spectrum, img

        # Envia o deslocamento do PAR (t_n-1, t_n)
        # Se prev_spectrum era None (primeira imagem), envia (0.0, 0.0)
        if prev_spectrum is not None:
            conn_out.send((dx_mm, dy_mm))
            acc_disp_local = [0.0, 0.0]  # Zera o local


class AsyncOdometerHooks:
    """
    Classe auxiliar para gerenciar os hooks (conexões e threads) do modo assíncrono.
    """

    def __init__(self, odometer_instance):
        self.odometer = odometer_instance
        self.accumulated_displacements = [0.0, 0.0]
        self.displacement_lock = threading.Lock()

        self._setup_async_mode()

    def _setup_async_mode(self):
        # Criação dos pipes
        self.pipe_main_to_pre_send, pipe_main_to_pre_recv = Pipe()
        pipe_pre_to_svd_send, pipe_pre_to_svd_recv = Pipe()
        pipe_svd_to_main_send, self.pipe_svd_to_main_recv = Pipe()

        # Criação dos processos workers
        self.proc_preprocess = Process(
            target=worker_img_preprocess,
            args=(pipe_main_to_pre_recv, pipe_pre_to_svd_send, self.odometer.configs),
            daemon=True,
        )
        self.proc_svd = Process(
            target=worker_svd,
            args=(
                pipe_pre_to_svd_recv, pipe_svd_to_main_send,
                self.odometer.configs, self.odometer.xres, self.odometer.yres
            ),
            daemon=True,
        )

        # Inicia os processos
        self.proc_preprocess.start()
        self.proc_svd.start()

        # Thread para receber deslocamentos dos workers
        self.result_thread = threading.Thread(target=self._displacement_receiver, daemon=True)
        self.result_thread.start()

        # Bind dos métodos assíncronos
        self.odometer.feed_image = self.feed_image_async
        self.odometer.get_displacement = self.get_displacement_async

    def _displacement_receiver(self):
        """Thread que recebe deslocamentos dos workers e os acumula."""
        while True:
            try:
                displacement = self.pipe_svd_to_main_recv.recv()
                if displacement is None:
                    break

                dx, dy = displacement

                with self.displacement_lock:
                    self.accumulated_displacements[0] += dx
                    self.accumulated_displacements[1] += dy
            except EOFError:
                # Pipe foi fechado
                break
            except Exception as e:
                print(f"Erro no receiver: {e}")
                break

    def feed_image_async(self, img):
        """Envia imagem para processamento assíncrono."""
        try:
            # Garante que a imagem seja contígua para melhor serialização
            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)
            self.pipe_main_to_pre_send.send(img)
        except Exception as e:
            pass
            #print(f"Erro ao enviar imagem: {e}")

    def get_displacement_async(self):
        """Obtém e zera os deslocamentos acumulados."""
        try:
            with self.displacement_lock:
                displacement = tuple(self.accumulated_displacements)
                self.accumulated_displacements = [0.0, 0.0]

            # Atualiza posição e contador
            if displacement != (0.0, 0.0):
                self.odometer.current_position += np.array(displacement)
                self.odometer.number_of_displacements += 1

            return displacement

        except Exception as e:
            print(f"Erro ao obter deslocamento: {e}")
            return 0.0, 0.0

    def shutdown(self):
        """Encerra os processos workers de forma limpa."""
        try:
            # Envia sinal de parada
            self.pipe_main_to_pre_send.send(None)

            # Aguarda término dos processos
            self.proc_preprocess.join(timeout=2)
            self.proc_svd.join(timeout=2)

            # Força término se necessário
            if self.proc_preprocess.is_alive():
                self.proc_preprocess.terminate()
                self.proc_preprocess.join(timeout=1)
            if self.proc_svd.is_alive():
                self.proc_svd.terminate()
                self.proc_svd.join(timeout=1)

            # Fecha os pipes
            self.pipe_main_to_pre_send.close()
            self.pipe_svd_to_main_recv.close()

        except Exception as e:
            print(f"Erro ao encerrar workers: {e}")