"""
    Example for processing the displacements two isolated images.
"""
from visual_odometer.lib.arraylib import xp_backend, set_backend
from visual_odometer import VisualOdometer
from PIL import Image, ImageOps
import time

def load_img(filename):
    img_array_rgb = Image.open(filename)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    return img_grayscale

grayscale_img0 = load_img('../datasets/dario_320x240/img.png') # image at t = t₀
grayscale_img1 = load_img('../datasets/dario_320x240/img_translated.png') # image at t = t₀ + Δt

set_backend(use_gpu=True) # Tenta usar gpu se disponível
xp = xp_backend()
img0 = xp.asarray(grayscale_img0)
img1 = xp.asarray(grayscale_img1)

odometer = VisualOdometer()
odometer.save_config('./')

odometer.calibrate(new_xres=1.0, new_yres=1.0)
t0 = time.time()
dx, dy = odometer.estimate_displacement_between(img0, img1)
dt = (time.time() - t0) * 1000

print(f'Displacement estimate: x = {dx}, y = {dy}')
print(f' dt = {dt:.3f} ms, fps = {1000/dt:.3f} Hz')