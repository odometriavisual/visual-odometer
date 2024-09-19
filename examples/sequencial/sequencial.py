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
try:
    odometer._start_pool()
    time.sleep(.1)

    for img in img_stream:
        odometer.feed_image(img)
        time.sleep(1 / fps)
        # odometer.get_displacement()

finally:
    odometer._reset_pool()


# Opção B de uso:
# with odometer:
#     time.sleep(.1)
#     for img in img_stream:
#         odometer.feed_image(img)
#         time.sleep(1 / fps)
#         # odometer.get_displacement()
