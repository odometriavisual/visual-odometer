import sys
import warnings

sys.path.insert(0, r'C:\Users\Panther\PycharmProjects\virtualencoder')

import cv2
import serial.tools.list_ports

from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
from virtualencoder.visualodometry.svd_decomposition import optimized_svd_method
from virtualencoder.visualodometry.dsp_utils import cv2_to_nparray_grayscale, image_preprocessing

import config

import signal
import time
from datetime import datetime, timedelta
import csv


def handler(signum, frame):
    print("Tempo esgotado!")
    raise Exception("Fim do tempo")




# Configurações para manter a conexão serial e ler dados
delta_keep_alive = timedelta(milliseconds=10)  # Intervalo entre as verificações de conexão
retry = 10  # Número de tentativas de comunicação antes de fechar a porta
connecting = True  # Indica se está tentando se conectar ao dispositivo

# Inicializa um arquivo CSV para salvar os dados
start_time = datetime.now()
last_call = start_time
current_time = start_time.strftime("%Y%m%d-%H%M%S")
filename = current_time + '.csv'
logging = open(filename, mode='a')
writer = csv.writer(logging, delimiter=",", escapechar=' ', quoting=csv.QUOTE_NONE)

print('Iniciando a conexão...')

signal.signal(signal.SIGALRM, handler)
serial_port_list = serial.tools.list_ports.comports()
selected_port = sorted(serial_port_list)[0]

resto_x = 0
resto_y = 0

intxacumulado = 0
intyacumulado = 0




print('Pegando acesso a camera, isso pode demorar um pouco...')
try:
    vid = cv2.VideoCapture(config.camera_id)
except:
    print("Erro ao configurar a câmera, por favor olhe o manual")
    exit()

try:
    ret, frame = vid.read()
    img_array = cv2_to_nparray_grayscale(frame)
except:
    print("Não foi possível conectar a câmera, altere o id da camera no config.py")
    exit()

if config.camera_exposure != None:
    print("Definindo exposição da câmera")
    vid.set(cv2.CAP_PROP_EXPOSURE, config.camera_exposure)

frame_num = -10 #Definido para negativo, o frame será contabilizado apenas após a décima imagem

total_deltax = 0
total_deltay = 0

M = None
N = None
start_time = 0


while True:
    try:
        time_now = datetime.now()
        while (connecting):
            try:
                print("Conectando...")
                ser = serial.Serial(port=str(selected_port))  # Formato ttyUSBx no Linux

                # Configurações da porta serial
                ser.baudrate = 115200  # Taxa de baud: 115200
                ser.bytesize = 8  # Bits de dados: 8
                ser.parity = 'N'  # Paridade: Nenhuma
                ser.stopbits = 1  # Bits de parada: 1
                ser.timeout = None  # Sem timeout, espera infinita

                time.sleep(0.05)
                connecting = False
                ser.flushInput()
                retry = 10
            except:
                pass

        # Verifica se é hora de enviar um sinal de "keep-alive"
        if ((time_now - last_call) > delta_keep_alive):
            last_call = time_now
            delta_keep_alive = timedelta(milliseconds=100)

            # Verifica se ainda há tentativas de comunicação
            if (retry > 0):
                ser.write(bytes('x', 'utf-8'))
                retry -= 1
            else:
                ser.close()
                connecting = True

        # Verifica se há dados disponíveis na porta serial
        serial_available = ser.in_waiting
        if (serial_available > 0):
            signal.alarm(1)
            ser_bytes = ser.readline()
            retry = 10
            signal.alarm(0)

            # Decodifica os bytes recebidos e registra o tempo atual
            decoded_bytes = (ser_bytes[0:len(ser_bytes) - 2].decode("utf-8"))
            current_time = datetime.now().strftime("%H:%M:%S.%f")

            # Exibe e escreve os dados recebidos no arquivo CSV
            print(current_time, decoded_bytes)
            writer.writerow([current_time, decoded_bytes, total_deltax, total_deltay])
            ser.flush()

        else:
            # Aguarda um curto período se não houver dados disponíveis
            print("Nenhum dado disponível.")
            time.sleep(0.5)

        ret, frame = vid.read()
        img_array = cv2_to_nparray_grayscale(frame)
        img_windowed = apply_border_windowing_on_image(img_array, config.border_windowing_method)
        img_processed = image_preprocessing(img_array)
        if frame_num > 0:
            if (M == None):
                print("Script iniciado")
                start_time = time.time()
                M, N = img_array.shape
            deltax, deltay = optimized_svd_method(img_processed, img_processed_old, M, N, phase_windowing=config.phase_windowing)
            multiplied_deltax = config.deltax_multiplier * deltax
            multiplied_deltay = config.deltay_multiplier * deltay
            total_deltax = total_deltax + multiplied_deltax
            total_deltay = total_deltay + multiplied_deltay


        frame_num = frame_num + 1
        img_processed_old = img_processed
    except KeyboardInterrupt:
        vid.release()

        passed_time = (time.time() - start_time)
        print("--- %s seconds ---" % passed_time)
        print("--- %s  frames ---" % frame_num)

        fps = frame_num / passed_time
        print("--- %s     fps ---" % fps)
        print("")
        print(f"Total deltax: {total_deltax}")
        print(f"Total deltay: {total_deltay}")

        exit()

        ser.close()
        logging.close()
        print("Registro concluído.")
    except Exception as exc:
        print("Erro:", exc)
    except:
        # Fecha a conexão serial e o arquivo de log em caso de erro
        ser.close()
        logging.close()
        print("Erro inesperado. Fechando conexão.")
        break