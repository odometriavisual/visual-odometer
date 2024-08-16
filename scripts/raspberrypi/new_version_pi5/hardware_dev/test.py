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
GPIO.setup(13, GPIO.IN)
GPIO.setup(18, GPIO.IN)

# Display Parameters
WIDTH = 128
HEIGHT = 64
BORDER = 5

# Variable pre-configuration
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(diretorio_atual, 'data')

# Verifica se o diretório "data" existe
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

arquivos_txt = [arquivo for arquivo in os.listdir(data_dir) if arquivo.endswith(".txt")]

# Status de exemplo
exemple = 0 
recording = False
if exemple == 0:
    defices = "ok"
elif exemple == 1:
    defices = "noCAM"
elif exemple == 2:
    defices = "noIMU"
elif exemple == 3:
    defices = "noPul"
else:
    defices = "..."

focus = 255

# Display Refresh

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

# Long text for scrolling
if arquivos_txt:
    long_text = arquivos_txt[0]
else:
    long_text = "No text files found."

# Calculate the width of the long text
text_bbox = draw.textbbox((0, 0), long_text, font=font)
text_width = text_bbox[2] - text_bbox[0]

# Calculate the width of the static text "last:"
last_text = "last:"
last_text_bbox = draw.textbbox((0, 0), last_text, font=font)
last_text_width = last_text_bbox[2] - last_text_bbox[0]

# Initial scroll position
scroll_position = 0
cont = 0

while True:

    if GPIO.input(18):
        focus = random.randint(0, 255)
        time.sleep(0.5)

    if GPIO.input(13):
        cont += 1
        if cont == 1:
            recording = True
            
        else:
            recording = False
            cont = 0
        time.sleep(0.5)

    # Draw a black filled box to clear the image.
    draw.rectangle((0, 0, oled.width, oled.height), outline=0, fill=0)

    # Obtaining system information
    IP = subprocess.check_output("hostname -I | cut -d' ' -f1", shell=True).decode().strip()
    Record = "Recording: yes" if recording else "Recording: no"
    conect = f"defices: {defices}"
    foco = f"focus: {focus}"

    # Draw the static system information
    draw.text((0, 2), f"IP: {IP}", font=font, fill=255)             # Line 1
    draw.text((0, 14), Record, font=font, fill=255)                 # Line 2
    draw.text((0, 26), conect, font=font, fill=255)                 # Line 3
    draw.text((0, 38), foco, font=font, fill=255)                   # Line 4 (static text)

    # Draw the static part "last:"
    draw.text((80, 38), last_text, font=font, fill=255)              # Line 5 (static part)

    # Draw the scrolling text part
    draw.text((80 + last_text_width - scroll_position, 50), long_text, font=font, fill=255)  # Line 5 (scrolling part)

    # Move scroll position
    scroll_position += 2  # Increase the scroll step size for faster scrolling
    if scroll_position > text_width:
        scroll_position = 0

    # Display image
    oled.image(image)
    oled.show()

    time.sleep(0.05)
