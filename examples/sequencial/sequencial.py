from visual_odometer import VisualOdometer
import time

def load(filename):
    from PIL import Image, ImageOps
    import numpy as np

    img_array_rgb = Image.open(filename)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    img_array = np.asarray(img_grayscale)

    return img_array


img0 = load('./img.png')  # image at t = t₀
img1 = load('./img_translated.png')  # image at t = t₀ + Δt

stream_size = 100
img_stream = [img0, img1] * stream_size

odometer = VisualOdometer(img_size=(640, 480))
fps = 60

time.sleep(1)

t0 = time.time()
for img in img_stream:
    odometer.feed_image(img)
    time.sleep(1 / fps)
    odometer.get_displacement()
delta_t = time.time() - t0

print(f"""
Number of frames: {len(img_stream)}
Processed frames: {odometer.number_of_displacements} out of {len(img_stream)}
Real FPS = {odometer.number_of_displacements / delta_t:.2f}."
      """)








