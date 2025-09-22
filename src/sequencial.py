"""
    Example for processing the displacements between frames in an image stream.
"""

from visual_odometer import VisualOdometer
import time
from glob import glob
from PIL import Image, ImageOps
import numpy as np
import os

def load_img(filename):
    img_array_rgb = Image.open(filename)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    return np.array(img_grayscale)

def main():
    filenames = glob('C:/Users/Demarky/Documents/Ensaios de referencia/Dario/TesteRapido/*.jpg')
    img_stream = [load_img(f) for f in filenames]

    odometer = VisualOdometer(img_shape=img_stream[0].shape, async_mode=False)

    time.sleep(1)
    t0 = time.time()
    acc_displacement = [0,0]
    all_displacements = []  # <- acumula deslocamentos
    timestamps = []         # <- guarda timestamps

    for img in img_stream:
        odometer.feed_image(img)
        time.sleep(0.5)
        displacement = odometer.get_displacement()
        acc_displacement[0] += displacement[0]
        acc_displacement[1] += displacement[1]

        all_displacements.append(displacement)
        timestamps.append(time.time() - t0)

    # Salvar TXT simples (fácil de colar no Sheets)
    txt_file = "displacements_test.txt"
    with open(txt_file, "w") as f:
        f.write("timestamp,dx,dy\n")  # cabeçalho (usa tabulação como separador)
        for t, (dx, dy) in zip(timestamps, all_displacements):
            f.write(f"{t:.3f},{dx:.3f},{dy:.3f}\n")

    print(f"✅ Deslocamentos salvos em {os.path.abspath(txt_file)}")

    # Estatísticas
    desired_displacement = [150, 179]
    delta_t = time.time() - t0
    delta_error = [abs(desired_displacement[0] - acc_displacement[0]),
                   abs(desired_displacement[1] - acc_displacement[1])]
    percentual_error = [100 * delta_error[0] / desired_displacement[0],
                        100 * delta_error[1] / desired_displacement[1]]
    erro_medio = (percentual_error[0] + percentual_error[1]) / 2
    print(f"erro_medio: {erro_medio:.2f}%")
    print(f'Accumulated displacement: {acc_displacement}')
    print(f"""
Number of frames: {len(img_stream)}
Processed frames: {odometer.number_of_displacements} out of {len(img_stream)}
Real FPS = {odometer.number_of_displacements / delta_t:.2f}.
          """)

    odometer.shutdown()


if __name__ == '__main__':
    main()
