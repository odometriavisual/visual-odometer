from flask import Flask, send_file, abort, render_template
import concurrent.futures
import threading
import os
import time
import datetime
import webbrowser

import sys
import warnings

sys.path.insert(0, r'C:\Users\Panther\PycharmProjects\virtualencoder')

import cv2
import serial.tools.list_ports

from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
from virtualencoder.visualodometry.svd_decomposition import optimized_svd_method
from virtualencoder.visualodometry.dsp_utils import cv2_to_nparray_grayscale, image_preprocessing

import config

import matplotlib.pyplot as plt
import numpy as np

import serial
import serial.tools.list_ports

import time

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Obtém o diretório atual do arquivo Python
dir_path = os.path.dirname(os.path.realpath(__file__))

# Caminho para o arquivo HTML
html_file_path = os.path.join(dir_path, 'executar.html')



app = Flask(__name__)
app.template_folder = ''

lock_quat = threading.Lock()
printar = False
lista_dados = []  # este é o Y
lista_dados2 = []  # este é o X
lista_imu = []
lista_3d = []

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(diretorio_atual, "data")
files = os.listdir(data_dir)
for file_name in files:
    file_path = os.path.join(data_dir, file_name)
    os.remove(file_path)

from virtualencoder.visualodometry.score_focus import score_teng

vid = cv2.VideoCapture(config.camera_id)
camera_id = 0  # Altere o id da câmera aqui
total_rec_time = 60  # seconds
max_fps = 30  # Define o FPS máximo desejado
camera_exposure = -6  # Defina exposição da câmera

score_history = [0] * 270
counter = 0

vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
vid.set(cv2.CAP_PROP_FOCUS, counter)

serial_encoder = 0

gyroData = [0,0,0,0]

# --- Funções reservadas para envio e armazenamento de dados --- #

def checkSerialInput():
    global gyroData, serial_giroscopio
    if (serial_giroscopio.in_waiting > 0):
        ser_line = serial_giroscopio.readline().decode()
        gyroData = [float(x) for x in ser_line.split(",")] #list  quaternium to biallhole
        serial_giroscopio.flush()

def serialSendEncoder(x, y):
    global resto_x, resto_y, serial_encoder, intxacumulado, intyacumulado, serial_encoder

    x += resto_x
    y += resto_y

    intx = int(x)
    inty = int(y)

    text_to_send = f"{intx},{inty},0\n"
    serial_encoder.write(text_to_send.encode())

    resto_x = x - intx
    resto_y = y - inty

# ----- Configuração de comunicação Serial ----- #

#Comentado para remover dados seriais

def minha_thread():
    global printar
    global vid
    global resto_x
    global resto_y
    global lista_imu
    global lista_3d
    global intxacumulado
    global intyacumulado
    global serial_encoder
    global serial_giroscopio

    totalx = 0
    totaly = 0
    giroscop = 0
    # --- definição de variáveis necessárias --- #

    total_deltax = 0
    total_deltay = 0

    M = None
    N = None
    start_time = 0

    resto_x = 0
    resto_y = 0

    intxacumulado = 0
    intyacumulado = 0

    ports = serial.tools.list_ports.comports()

    for port in ports:
        print(port)
        if port.serial_number == "5698028262":
            print("Iniciando conecção com o modulo do giroscópio")
            serial_giroscopio = serial.Serial(port=port.device, baudrate=115200, timeout=1)
            serial_giroscopio.setRTS(False)
            time.sleep(0.3)
            serial_giroscopio.setRTS(True)
            time.sleep(0.3)
            serial_giroscopio.setRTS(False)
            time.sleep(0.3)
        elif port.serial_number == "5698010135":
            print("Iniciando comunicação com o modulo pulsador")
            serial_encoder = serial.Serial(port=port.device, baudrate=115200, timeout=1)
        print("TESTE")

    print("Testando comunicação serial: encoder")
    _ = serial_encoder.read()
    print("Testando comunicação serial: giroscópio server")
    _ = serial_giroscopio.read()
    print("Comunicação serial OK")
    print("")

    # -- Inicio da configuração da câmera --- #

    print('Pegando acesso a camera, isso pode demorar um pouco...')
    vid = cv2.VideoCapture(config.camera_id)

    ret, frame = vid.read()
    img_array = cv2_to_nparray_grayscale(frame)
    if config.camera_exposure != None:
        print("Definindo exposição da câmera")
        vid.set(cv2.CAP_PROP_EXPOSURE, config.camera_exposure)

    # Abre o arquivo HTML no navegador padrão
    webbrowser.open('file://' + html_file_path)

    frame_num = -10
    while True:
        try:
            # Comentado para remover serial
            checkSerialInput()

            ret, frame = vid.read()
            img_array = cv2_to_nparray_grayscale(frame)
            img_windowed = apply_border_windowing_on_image(img_array, config.border_windowing_method)
            img_processed = image_preprocessing(img_array)
            if frame_num > 0:
                if (M == None):
                    print("Script iniciado")
                    start_time = time.time()
                    M, N = img_array.shape
                deltax, deltay = optimized_svd_method(img_processed, img_processed_old, M, N,
                                                      phase_windowing=config.phase_windowing)
                multiplied_deltax = config.deltax_multiplier * deltax
                multiplied_deltay = config.deltay_multiplier * deltay

                # Comentado para remover serial
                serialSendEncoder(multiplied_deltax,
                                  multiplied_deltay)  # <- Envia informações de deslocamento para o modulo pulsador

                total_deltax = total_deltax + multiplied_deltax
                total_deltay = total_deltay + multiplied_deltay

                # Exemplo de array para ser salvo:
                array_to_save = [time.time(), gyroData, total_deltax, total_deltay]
                print(array_to_save)

                # salvando x e y para o web
                # Recomendo limitar o salvamento a um intervalo. Mas é só uma sugestão mesmo.

                totaly = round(total_deltay, 2)
                totalx = round(total_deltax, 2)

                r = R.from_quat([gyroData])
                v = [totalx, totaly, 0]
                ddd = r.apply(v)

                lock_quat.acquire()
                if printar:
                    lista_dados2.append(totalx)
                    lista_dados.append(totaly)
                    lista_imu.append(gyroData)
                    lista_3d.append(ddd)
                lock_quat.release()



            frame_num = frame_num + 1
            img_processed_old = img_processed

        except KeyboardInterrupt:

            # Fechando portas Seriais
            serial_giroscopio.setRTS(True)
            serial_encoder.setRTS(True)
            time.sleep(0.3)
            serial_giroscopio.setRTS(False)
            serial_encoder.setRTS(False)
            time.sleep(0.3)
            serial_giroscopio.close()
            serial_encoder.close()
            # -----------------------#

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
        except Exception as exc:
            print("Erro:", exc)


def salvar_dados_arquivo():
    global lista_dados, lista_dados2, lista_imu
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Cria um timestamp único
    arquivo_dados = os.path.join(data_dir, f"dados_{timestamp}.txt")  # Nome do arquivo com timestamp
    with open(arquivo_dados, "w") as arquivo:
        for x, y, z in zip(lista_dados, lista_dados2, lista_imu):
            arquivo.write(
                f"{x}|{y}|{z}\n")  # Escreve os dados x e y em uma linha, separados por um espaço e com uma quebra de linha no final


@app.route('/iniciar', methods=["GET", "POST"])
def iniciar():
    global printar

    printar = True
    return '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;"><form action="/3D" method="post"><button type="submit" style="width: 150px; height: 50px;">3D</button></form><form action="/finalizar" method="post"><button type="submit" style="width: 150px; height: 50px;">stop</button></form></div>'

@app.route('/3D', methods=["GET", "POST"])
def vizu3d():
    global lista_3d
    # Dados
    dados = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    # Extrair valores x, y, z do vetor
    x = dados[:, 0]
    y = dados[:, 1]
    z = dados[:, 2]

    # Passar os dados para o template HTML
    return render_template('index.html', x=x.tolist(), y=y.tolist(), z=z.tolist())


@app.route('/finalizar', methods=["GET", "POST"])
def finalizar():
    print("teste")
    global printar
    global lista_dados, lista_dados2, lista_imu, lista_3d
    salvar_dados_arquivo()
    lista_dados = []  # Limpa a lista após salvar os dados
    lista_dados2 = []
    lista_imu = []
    lista_3d = []
    printar = False
    return '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;"><form action="/iniciar" method="post"><button type="submit" style="width: 150px; height: 50px;">start</button></form><br><form action="/dados" method="post"><button type="submit" style="width: 150px; height: 50px;">results</button></form></div>'


@app.route('/dados', methods=["GET", "POST"])
def mostrar_dados():
    html = '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;">'

    # Listar todos os arquivos .txt no diretório atual
    arquivos_txt = [arquivo for arquivo in os.listdir(data_dir) if arquivo.endswith(".txt")]

    # Criar botões para cada arquivo .txt
    for arquivo in arquivos_txt:
        html += f'<form action="/abrir_arquivo/{arquivo}" method="post"><button type="submit" style="width: 150px; height: 50px;">{arquivo}</button></form>'

    html += '<br><br><form action="/iniciar" method="post"><button type="submit" style="width: 150px; height: 50px;">start</button></form></div>'
    return html


@app.route('/abrir_arquivo/<nome_arquivo>', methods=["GET", "POST"])
def abrir_arquivo(nome_arquivo):
    try:
        arquivo_path = os.path.join(data_dir, nome_arquivo)
        if os.path.isfile(arquivo_path):
            with open(arquivo_path, "r") as arquivo:
                conteudo = arquivo.read()
            return f'<div style="white-space: pre-line;">{conteudo}<script>window.location.href = "http://localhost:5000/download/{nome_arquivo}"</script></div>'
        else:
            abort(
                404)  # Retorna um erro 404 se o arquivo não existir  <form action="/download" method="post"><button type="submit" style="width: 150px; height: 50px;">download</button>

    except Exception as e:

        return "Erro interno do servidor", 500  # Retorna um erro 500 se ocorrer uma exceção


@app.route('/download/<nome_arquivo>', methods=["GET"])
def download(nome_arquivo):
    try:
        arquivo_path = os.path.join(data_dir, nome_arquivo)
        if os.path.isfile(arquivo_path):
            return send_file(arquivo_path, as_attachment=True)
        else:
            abort(404)  # Retorna um erro 404 se o arquivo não existir
    except Exception as e:
        return "Erro interno do servidor", 500  # Retorna um erro 500 se ocorrer uma exceção


@app.route('/autofoco', methods=["GET", "POST"])
def autofoco():
    global vid
    global counter
    global score_history

    while counter < 260:
        ret, frame = vid.read()
        cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        focus_score = score_teng(cv2_img)
        score_history[counter] = focus_score
        counter += 5
        vid.set(cv2.CAP_PROP_FOCUS, counter)

    vid.set(cv2.CAP_PROP_FOCUS, np.argmax(score_history))
    return f'<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;">{np.argmax(score_history)}<form action="/iniciar" method="post"><button type="submit" style="width: 150px; height: 50px;">start</button></form><br><form action="/dados" method="post"><button type="submit" style="width: 150px; height: 50px;">results</button></form></div>'


if __name__ == "__main__":
    thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    thread_executor.submit(minha_thread)
    app.run(host="127.0.0.1", port=5000)
