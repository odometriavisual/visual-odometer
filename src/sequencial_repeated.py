"""
    Example for processing the displacements between frames in an image stream.
"""

from visual_odometer import VisualOdometer
import time
from glob import glob
from PIL import Image, ImageOps
import numpy as np

def load_img(filename):
    img_array_rgb = Image.open(filename)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    return np.array(img_grayscale)

def main():
    filenames = glob('../datasets/dario_320x240/*.png')
    img0 = load_img(filenames[0])
    img1 = load_img(filenames[1])

    stream_size = 100
    img_stream = [img0, img1] * stream_size

    odometer = VisualOdometer(img_shape=img0.shape, async_mode=True)

    # Primeiras imagens para inicializar
    odometer.feed_image(img0)
    odometer.get_displacement()
    odometer.feed_image(img1)
    odometer.get_displacement()

    time.sleep(1)  # Aguarda worker inicializar

    t0 = time.time()

    for img in img_stream:
        odometer.feed_image(img)
        displacement = odometer.get_displacement()
        print(displacement)

    delta_t = time.time() - t0

    print(f"""
Number of frames: {len(img_stream)}
Processed frames: {odometer.number_of_displacements} out of {len(img_stream)}
Real FPS = {odometer.number_of_displacements / delta_t:.2f}."
          """)

    # Importante: encerrar o worker
    odometer.shutdown()

if __name__ == '__main__':
    main()