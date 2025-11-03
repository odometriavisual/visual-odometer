"""
    Example for processing the displacements two isolated images.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from visual_odometer import VisualOdometer
from PIL import Image, ImageOps
import numpy as np
import time

def load_img(filename):
    img_array_rgb = Image.open(filename)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    return img_grayscale

img0 = np.asarray(load_img('../datasets/dario_320x240/img.png')) # image at t = t₀
img1 = np.asarray(load_img('../datasets/dario_320x240/img_translated.png')) # image at t = t₀ + Δt

odometer = VisualOdometer(img_shape=img0.shape)

t0 = time.time()
dx, dy = odometer.estimate_displacement_between(img0, img1)
dt = (time.time() - t0) * 1000

print(f'Displacement estimate: x = {dx}, y = {dy}')
print(f' dt = {dt:.3f} ms, fps = {1000/dt:.3f} Hz')
