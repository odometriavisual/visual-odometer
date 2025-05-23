from visual_odometer import VisualOdometer
import time
from PIL import Image, ImageOps
import numpy as np

use_gpu = True
try:
    import cupy as cp
except:
    use_gpu = False
    pass

def load_img(filename):
    img_array_rgb = Image.open(filename)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    return img_grayscale

grayscale_img0 = load_img('./img.png')  # image at t = t₀
grayscale_img1 = load_img('./img_translated.png') # image at t = t₀ + Δt

if use_gpu:
    img0 = cp.asarray(grayscale_img0)
    img1 = cp.asarray(grayscale_img1)
else:
    img0 = np.asarray(grayscale_img0)
    img1 = np.asarray(grayscale_img1)

stream_size = 100
img_stream = [img0, img1] * stream_size

odometer = VisualOdometer(img_size=(640, 480))
odometer.feed_image(img0)
odometer.get_displacement()
odometer.feed_image(img1)
odometer.get_displacement()

time.sleep(1)

t0 = time.time()

for img in img_stream:
    #load_as_cpu_img('./img.png')
    odometer.feed_image(img)
    odometer.get_displacement()
delta_t = time.time() - t0

print(f"""
Number of frames: {len(img_stream)}
Processed frames: {odometer.number_of_displacements} out of {len(img_stream)}
Real FPS = {odometer.number_of_displacements / delta_t:.2f}."
      """)








