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
        # Attempt to unmount the USB drive if it is mounted
        try:
            command = ["sudo", "umount", "/media/my32gb"]
            subprocess.run(command, check=True)
            print("USB drive successfully unmounted.")
        except subprocess.CalledProcessError:
            print("USB drive was not mounted or error while unmounting.")

        # Run the lsblk command to list block devices
        output = subprocess.check_output(['lsblk', '-o', 'NAME,TYPE'], text=True)
        lines = output.strip().split('\n')

        # Identify the USB device
        for line in lines:
            if 'disk' in line:
                name = line.split()[0]
                device = "/dev/" + name
                print(f"USB device found: {device}")

                # Mount the device
                command = ["sudo", "mount", "-t", "vfat", "-o", "rw", device, "/media/my32gb"]
                try:
                    subprocess.run(command, check=True)
                    print("USB drive successfully mounted.")

                    # Adjust permissions of the mount directory
                    subprocess.run(["sudo", "chmod", "777", "/media/my32gb"], check=True)

                    return "/media/my32gb"
                except subprocess.CalledProcessError as e:
                    print(f"Error mounting USB drive: {e}")
                    return None

        print("No USB device found.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing lsblk: {e}")
        return None

# Function to copy the last saved file to the USB drive
def copy_file_to_usb(src_file, dest_dir):
    if not os.path.exists(src_file):
        print(f"File {src_file} not found.")
        return

    if not os.path.isdir(dest_dir):
        print(f"Destination directory {dest_dir} not found.")
        return

    # Accessing the data folder
    dest_file = os.path.join(dest_dir, os.path.basename(src_file))

    try:
        # Copy the file to the USB drive using sudo
        command = ["sudo", "cp", src_file, dest_file]
        subprocess.run(command, check=True)
        print(f"File {src_file} successfully copied to {dest_file}.")
    except subprocess.CalledProcessError as e:
        print(f"Error copying file: {e}")

# Function that displays the IP
def PlotIP():
    # Getting the IP
    IP = subprocess.check_output("hostname -I | cut -d' ' -f1", shell=True).decode().strip()
    # Clearing the IP line
    draw.rectangle((0, 2, oled.width, oled.height), outline=0, fill=0)
    # Displaying the IP
    draw.text((0, 2), f"IP: {IP}", font=font, fill=255)

# Display the last acquisition
def PlotLast():
    # Variable pre-configuration
    diretorio_atual = os.path.dirname(os.path.abspath(_file_))
    data_dir = os.path.join(diretorio_atual, 'data')

    # Check if the "data" directory exists
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    arquivos_txt = [arquivo for arquivo in os.listdir(data_dir) if arquivo.endswith(".txt")]

    if arquivos_txt:
        long_text = arquivos_txt[0]
    else:
        long_text = "No text files found."
        
    # Clear the line
    draw.rectangle((0, 50, oled.width, oled.height), outline=0, fill=0)
    # Write the updated line 
    draw.text((0, 50), long_text, font=font, fill=255)  # Line 5 (scrolling part)

# Function that shows if an acquisition is being made
# This function checks if "condition" is True or False
def PlotRecord(condition):
    draw.rectangle((0, 14, oled.width, 24), outline=0, fill=0)
    if condition:
        draw.text((0, 14), "Recording: yes", font=font, fill=255)
    else:
        draw.text((0, 14), "Recording: no", font=font, fill=255)

# Function that checks if all peripherals are connected, now with the IMU and camera 
# coming from the Raspberry Pi Zero, I believe there will only be 2 conditions
def PlotConect(condition):
    if condition == 0:
        devices = "ok"
    elif condition == 1:
        devices = "noCAM"
    elif condition == 2:
        devices = "noIMU"
    elif condition == 3:
        devices = "noPul"
    else:
        devices = "..."
    draw.rectangle((0, 26, oled.width, oled.height), outline=0, fill=0)
    draw.text((0, 26), f"devices: {devices}", font=font, fill=255)

# Function that displays the focus
def PlotFocus(condition):
    draw.rectangle((0, 38, 0 + oled.width, 48), outline=0, fill=0)
    draw.text((80, 38), "last:", font=font, fill=255)
    draw.text((0, 38), f"focus: {condition}", font=font, fill=255)

recording = True
focus = 0
cont = 0
# Calling all functions to initialize the display
PlotIP()
PlotRecord(recording)
PlotConect(0)
PlotFocus(focus)
PlotLast()

while True:
    # Check if the button was pressed to save the file to the USB drive
    if GPIO.input(18):
        # Find the data directory and the file to be saved
        diretorio_atual = os.path.dirname(os.path.abspath(_file_))
        data_dir = os.path.join(diretorio_atual, 'data')
        src_files = [arquivo for arquivo in os.listdir(data_dir) if arquivo.endswith(".txt")]

        # Find and mount the USB drive
        usb_path = find_usb_drive()
        if usb_path:
            # Copy each file to the USB drive
            for src_file in src_files:
                src_path = os.path.join(data_dir, src_file)
                copy_file_to_usb(src_path, usb_path)

            # Unmount the USB drive after copying
            try:
                command = ["sudo", "umount", usb_path]
                subprocess.run(command, check=True)
                print("USB drive successfully unmounted.")
            except subprocess.CalledProcessError as e:
                print(f"Error unmounting USB drive: {e}")

        time.sleep(0.5)
    
    # Check if the button was pressed to record the focus; currently generating a random number 
    # between 0 and 255 just for demonstration
    if GPIO.input(12):
        focus = random.randint(0, 255)
        time.sleep(0.5)
        PlotFocus(focus)
    
    # Check if the button was pressed to show if the trial is being recorded
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
