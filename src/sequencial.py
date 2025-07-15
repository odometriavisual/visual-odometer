"""
    Example for processing the displacements between frames in an image stream.
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

stream_size = 100
img_stream = [img0, img1] * stream_size
odometer = VisualOdometer()

odometer.feed_image(img0)
odometer.get_displacement()
odometer.feed_image(img1)
odometer.get_displacement()

time.sleep(1)

t0 = time.time()

for img in img_stream:
    odometer.feed_image(img)
    print(odometer.get_displacement())
delta_t = time.time() - t0

print(f"""
Number of frames: {len(img_stream)}
Processed frames: {odometer.number_of_displacements} out of {len(img_stream)}
Real FPS = {odometer.number_of_displacements / delta_t:.2f}."
      """)
