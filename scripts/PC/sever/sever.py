from flask import Flask, send_file, abort
import concurrent.futures
import threading
import os
import time
import datetime

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


resto_x = 0
resto_y = 0

intxacumulado = 0
intyacumulado = 0

app = Flask(__name__)

lock_quat = threading.Lock()
printar = False
lista_dados = []  #este é o Y
lista_dados2 = []  #este é o X

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
arquivo_dados = os.path.join(diretorio_atual, "dados.txt")


from virtualencoder.visualodometry.score_focus import score_teng
vid = cv2.VideoCapture(config.camera_id)
camera_id = 0  # Altere o id da câmera aqui
total_rec_time = 60  # seconds
max_fps = 30  # Define o FPS máximo desejado
camera_exposure = -6  # Defina exposição da câmera

score_history = [0]*270
counter = 0

vid.set(cv2.CAP_PROP_AUTOFOCUS, 0)
vid.set(cv2.CAP_PROP_FOCUS, counter)

# def atualizarPos(x,y):
#     global resto_x, resto_y, arduino, intxacumulado, intyacumulado
#
#     x += resto_x
#     y += resto_y
#
#     intx = int(x)
#     inty = int(y)
#
#     text_to_send = f"0,{intx}, {inty} \n"
#     arduino.write(text_to_send.encode())
#     #_ = arduino.readline()
#
#     resto_x = x - intx
#     resto_y = y - inty
#
# if config.usb_com_port is None:
#     try:
#         print("Iniciando setup automático de comunicação Serial")
#         serial_port_list = serial.tools.list_ports.comports()
#         serial_port_list_size = len(serial_port_list)
#         if (serial_port_list_size == 0):
#             print ("Não foi detectado nenhuma comunicação serial compatível")
#         elif (serial_port_list_size > 1):
#             warnings.warn("ATENÇÃO - Foram encontradas mais de uma porta serial, o código exercutaa apenas com uma delas")
#         selected_port = sorted(serial_port_list)[0]
#         arduino = serial.Serial(port=selected_port.name, baudrate=115200, timeout=1)
#         state = arduino.read()
#         print(f"Porta {selected_port.name} conectada")
#     except:
#         print("Erro na conexão da comunicação serial, verifique se o encoder está conectado")
#         exit()
# else:
#     try:
#         arduino = serial.Serial(port=config.usb_com_port, baudrate=115200, timeout=1)
#     except:
#         print("Erro na conexão da comunicação serial, é recomendado alterar a variável usb_com_port no config.py para None")
#         exit()


# - Fim do setup Serial - #

def minha_thread():
    global printar
    global vid
    total_deltax = 0
    total_deltay = 0
    totalx = 0
    totaly = 0
    while True:
        print('Pegando acesso a camera, isso pode demorar um pouco...')

        try:
            ret, frame = vid.read()
            img_array = cv2_to_nparray_grayscale(frame)
        except:
            print("Não foi possível conectar a câmera, altere o id da camera no config.py")
            exit()

        if config.camera_exposure != None:
            print("Definindo exposição da câmera")
            vid.set(cv2.CAP_PROP_EXPOSURE, config.camera_exposure)

        frame_num = -10  # Definido para negativo, o frame será contabilizado apenas após a décima imagem



        M = None
        N = None
        start_time = 0

        while True:
            try:
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
                    # atualizarPos(multiplied_deltax,multiplied_deltay)
                    total_deltax = total_deltax + (multiplied_deltax)
                    total_deltay = total_deltay + (multiplied_deltay)


                    print(
                        f"Frame:  {frame_num:>3.2f}, delta:[{deltax:>5.2f},{deltay:>5.2f}], total_delta:[{total_deltax:>5.2f}, {total_deltay:>5.2f}]")

                    totaly = round(total_deltay, 2)
                    totalx = round(total_deltax, 2)
                    lock_quat.acquire()
                    if printar:
                        lista_dados2.append(totalx)
                        lista_dados.append(totaly)
                    lock_quat.release()

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

def salvar_dados_arquivo():
    global lista_dados, lista_dados2
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Cria um timestamp único
    arquivo_dados = os.path.join(diretorio_atual, f"dados_{timestamp}.txt")  # Nome do arquivo com timestamp
    with open(arquivo_dados, "w") as arquivo:
        for x, y in zip(lista_dados, lista_dados2):
            arquivo.write(f"{x} | {y}\n")  # Escreve os dados x e y em uma linha, separados por um espaço e com uma quebra de linha no final

@app.route('/iniciar', methods=["GET", "POST"])
def iniciar():
    global printar

    printar = True
    return '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;"><form action="/finalizar" method="post"><button type="submit" style="width: 150px; height: 50px;">stop</button></form></div>'

@app.route('/finalizar', methods=["GET", "POST"])
def finalizar():
    global printar
    global lista_dados
    salvar_dados_arquivo()
    lista_dados = []  # Limpa a lista após salvar os dados
    printar = False
    return '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;"><form action="/iniciar" method="post"><button type="submit" style="width: 150px; height: 50px;">start</button></form><br><form action="/dados" method="post"><button type="submit" style="width: 150px; height: 50px;">results</button></form></div>'


@app.route('/dados', methods=["GET", "POST"])
def mostrar_dados():
    html = '<div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh;">'

    # Listar todos os arquivos .txt no diretório atual
    arquivos_txt = [arquivo for arquivo in os.listdir(diretorio_atual) if arquivo.endswith(".txt")]

    # Criar botões para cada arquivo .txt
    for arquivo in arquivos_txt:
        html += f'<form action="/abrir_arquivo/{arquivo}" method="post"><button type="submit" style="width: 150px; height: 50px;">{arquivo}</button></form>'

    html += '<br><br><form action="/iniciar" method="post"><button type="submit" style="width: 150px; height: 50px;">start</button></form></div>'
    return html


@app.route('/abrir_arquivo/<nome_arquivo>', methods=["GET", "POST"])
def abrir_arquivo(nome_arquivo):
    try:
        arquivo_path = os.path.join(diretorio_atual, nome_arquivo)
        if os.path.isfile(arquivo_path):
            with open(arquivo_path, "r") as arquivo:
                conteudo = arquivo.read()
            return f'<div style="white-space: pre-line;">{conteudo}</div>'
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