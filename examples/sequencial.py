import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from visual_odometer import VisualOdometer
import time
from glob import glob
from os import path
from PIL import Image, ImageOps
import numpy as np


def load_img(filename):
    img_array_rgb = Image.open(filename)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    return img_grayscale

img_stream = [(path.split(img_path)[1], np.asarray(load_img(img_path))) for img_path in glob('../datasets/dario_320x240/*.png')]

odometer = VisualOdometer(img_shape=img_stream[0][1].shape)
odometer.feed_image(img_stream[0][1])
odometer.get_displacement()
odometer.feed_image(img_stream[1][1])
odometer.get_displacement()

t0 = time.time()

for img_path, img in img_stream:
    ti0 = time.time()
    odometer.feed_image(img)
    odometer.get_displacement()
    ti1 = time.time()
    print(f'{img_path}: dt={(ti1-ti0)*1000:.3f}ms')

t1 = time.time()

delta_t = t1 - t0

print(f"""
Number of frames: {len(img_stream)}
Processed frames: {odometer.number_of_displacements} out of {len(img_stream)}
Real FPS = {odometer.number_of_displacements / delta_t:.2f}"
""")

