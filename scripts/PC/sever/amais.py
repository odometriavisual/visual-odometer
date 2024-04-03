from flask import Flask, send_file, abort
import concurrent.futures
import threading
import os
import time
import datetime

app = Flask(__name__)

lock_quat = threading.Lock()
printar = False
lista_dados = []
lista_dados2 = []

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
arquivo_dados = os.path.join(diretorio_atual, "dados.txt")


def minha_thread():
    global printar
    i = 0
    j = 0
    while True:
        lock_quat.acquire()
        if printar:
            lista_dados.append(i)
            lista_dados2.append(j)
        lock_quat.release()

        print(i)
        print(j)
        j -= 1
        i += 1
        time.sleep(1)


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




if __name__ == "__main__":
    thread_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    thread_executor.submit(minha_thread)
    app.run(host="127.0.0.1", port=5000)