import numpy as np

def ideal_lowpass(I, factor: float = 0.6):
    m = factor * I.shape[0] / 2
    n = factor * I.shape[1] / 2
    N = np.min(np.array([m, n]))
    N_val = int(N)
    I = I[int(I.shape[0] // 2 - N_val): int(I.shape[0] // 2 + N_val),
        int(I.shape[1] // 2 - N_val): int(I.shape[1] // 2 + N_val)]
    return I


# Spatial Windows:

def apply_raised_cosine_window(image):
    rows, cols = image.shape
    i = np.arange(rows)
    j = np.arange(cols)
    window = 0.5 * (1 + np.cos(np.pi * (2 * i[:, None] - rows) / rows)) * \
             0.5 * (1 + np.cos(np.pi * (2 * j - cols) / cols))
    return image * window

def blackman_harris_window(size: int, a0: float, a1: float, a2: float, a3: float):
    n = np.arange(size)
    window = (a0
              - a1 * np.cos(2 * np.pi * n / (size - 1))
              + a2 * np.cos(4 * np.pi * n / (size - 1))
              - a3 * np.cos(6 * np.pi * n / (size - 1)))
    return window

def apply_blackman_harris_window(image,
                                 a0: float = 0.35875, a1: float = 0.48829,
                                 a2: float = 0.14128, a3: float = 0.01168):

    height, width = image.shape
    window_row = blackman_harris_window(width, a0, a1, a2, a3, use_gpu=use_gpu)
    window_col = blackman_harris_window(height, a0, a1, a2, a3, use_gpu=use_gpu)
    image_windowed = np.outer(window_col, window_row) * image
    return image_windowed

def crop_two_imgs_with_displacement(imgA, imgB, dx, dy):
    h, w = imgA.shape

    # Corte no eixo x (invertido)
    if dx > 0:
        imgA = imgA[:, :w - dx]
        imgB = imgB[:, dx:]
    elif dx < 0:
        dx = abs(dx)
        imgA = imgA[:, dx:]
        imgB = imgB[:, :w - dx]

    # Corte no eixo y (invertido)
    if dy > 0:
        imgA = imgA[:h - dy, :]
        imgB = imgB[dy:, :]
    elif dy < 0:
        dy = abs(dy)
        imgA = imgA[dy:, :]
        imgB = imgB[:h - dy, :]

    # Garante que as imagens finais tenham o mesmo tamanho
    min_h = min(imgA.shape[0], imgB.shape[0])
    min_w = min(imgA.shape[1], imgB.shape[1])
    imgA = imgA[:min_h, :min_w]
    imgB = imgB[:min_h, :min_w]

    return imgA, imgB