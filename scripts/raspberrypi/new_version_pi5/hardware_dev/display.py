import time
import board
import busio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306
import os
import subprocess
import RPi.GPIO as GPIO
import random

# Cleanup GPIO
GPIO.cleanup()

# Set GPIO mode
GPIO.setmode(GPIO.BOARD)

# Buttons parameters
GPIO.setup(12, GPIO.IN)
GPIO.setup(13, GPIO.IN)
GPIO.setup(18, GPIO.IN)

# Display Parameters
WIDTH = 128
HEIGHT = 64
BORDER = 5

# Use for I2C.
i2c = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=0x3C)

# Clear display.
oled.fill(0)
oled.show()

# Create blank image for drawing.
image = Image.new("1", (oled.width, oled.height))
draw = ImageDraw.Draw(image)

# Draw a white background
draw.rectangle((0, 0, oled.width, oled.height), outline=255, fill=255)

# Load default font.
font = ImageFont.load_default()

def find_usb_drive():
    try:
        # Tenta desmontar o pendrive se estiver montado
        try:        
            command = ["sudo", "umount", "/media/my32gb"]
            subprocess.run(command, check=True)
            print("Pendrive desmontado com sucesso.")
        except subprocess.CalledProcessError:
            print("Pendrive não estava montado ou erro ao desmontar.")

        # Executa o comando lsblk para listar os dispositivos de bloco
        output = subprocess.check_output(['lsblk', '-o', 'NAME,TYPE'], text=True)
        lines = output.strip().split('\n')

        # Identifica o dispositivo USB
        for line in lines:
            if 'disk' in line:
                name = line.split()[0]
                device = "/dev/" + name
                print(f"Dispositivo USB encontrado: {device}")

                # Monta o dispositivo
                command = ["sudo", "mount", "-t", "vfat", "-o", "rw", device, "/media/my32gb"]
                try:
                    subprocess.run(command, check=True)
                    print("Pendrive montado com sucesso.")
                    
                    # Ajusta as permissões do diretório de montagem
                    subprocess.run(["sudo", "chmod", "777", "/media/my32gb"], check=True)
                    
                    return "/media/my32gb"
                except subprocess.CalledProcessError as e:
                    print(f"Erro ao montar o pendrive: {e}")
                    return None

        print("Nenhum dispositivo USB encontrado.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar lsblk: {e}")
        return None

def copy_file_to_usb(src_file, dest_dir):
    if not os.path.exists(src_file):
        print(f"Arquivo {src_file} não encontrado.")
        return

    if not os.path.isdir(dest_dir):
        print(f"Diretório de destino {dest_dir} não encontrado.")
        return

    dest_file = os.path.join(dest_dir, os.path.basename(src_file))

    try:
        # Copia o arquivo para o pendrive usando sudo
        command = ["sudo", "cp", src_file, dest_file]
        subprocess.run(command, check=True)
        print(f"Arquivo {src_file} copiado para {dest_file} com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao copiar o arquivo: {e}")

def PlotIP():
    IP = subprocess.check_output("hostname -I | cut -d' ' -f1", shell=True).decode().strip()
    draw.rectangle((0, 2, oled.width, oled.height), outline=0, fill=0)
    draw.text((0, 2), f"IP: {IP}", font=font, fill=255) 

def PlotLast():
    # Variable pre-configuration
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(diretorio_atual, 'data')

    # Verifica se o diretório "data" existe
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    arquivos_txt = [arquivo for arquivo in os.listdir(data_dir) if arquivo.endswith(".txt")]

    if arquivos_txt:
        long_text = arquivos_txt[0]
    else:
        long_text = "No text files found."
    
    draw.rectangle((0, 50, oled.width, oled.height), outline=0, fill=0)
    draw.text((0, 50), long_text, font=font, fill=255)  # Line 5 (scrolling part)

def PlotRecord(condition):
    draw.rectangle((0, 14, oled.width, 24), outline=0, fill=0)
    if condition:
        draw.text((0, 14), "Recording: yes", font=font, fill=255) 
    else:
        draw.text((0, 14), "Recording: no", font=font, fill=255) 

def PlotConect(condition):
    if condition == 0:
        defices = "ok"
    elif condition == 1:
        defices = "noCAM"
    elif condition == 2:
        defices = "noIMU"
    elif condition == 3:
        defices = "noPul"
    else:
        defices = "..."
    draw.rectangle((0, 26, oled.width, oled.height), outline=0, fill=0)
    draw.text((0, 26), f"defices: {defices}", font=font, fill=255)

def PlotFocus(condition):
    draw.rectangle((0, 38, 0 + oled.width, 48), outline=0, fill=0)
    draw.text((80, 38), "last:", font=font, fill=255) 
    draw.text((0, 38), f"focus: {condition}", font=font, fill=255)    

recording = True
focus = 0
cont = 0

PlotIP()
PlotRecord(recording)
PlotConect(0)
PlotFocus(focus)
PlotLast()

while True:
    if GPIO.input(18):
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(diretorio_atual, 'data')
        src_files = [arquivo for arquivo in os.listdir(data_dir) if arquivo.endswith(".txt")]

        # Encontra e monta o pendrive
        usb_path = find_usb_drive()
        if usb_path:
            # Copia cada arquivo para o pendrive
            for src_file in src_files:
                src_path = os.path.join(data_dir, src_file)
                copy_file_to_usb(src_path, usb_path)
            
            # Desmonta o pendrive após a cópia
            try:
                command = ["sudo", "umount", usb_path]
                subprocess.run(command, check=True)
                print("Pendrive desmontado com sucesso.")
            except subprocess.CalledProcessError as e:
                print(f"Erro ao desmontar o pendrive: {e}")
        # focus = random.randint(0, 255)
        time.sleep(0.5)
        # PlotFocus(focus)

    if GPIO.input(12):
        focus = random.randint(0, 255)
        time.sleep(0.5)
        PlotFocus(focus)

    if GPIO.input(13):
        cont += 1
        if cont == 1:
            recording = True
            PlotRecord(recording)
        else:
            recording = False
            PlotRecord(recording)
            PlotLast()
            cont = 0
        time.sleep(0.5)

    oled.image(image)
    oled.show()

    time.sleep(0.05)




