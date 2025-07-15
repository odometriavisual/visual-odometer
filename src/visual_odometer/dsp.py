from .lib.arraylib import xp_backend

# Frequency Windows:
def ideal_lowpass(I, factor: float = 0.6):
    xp = xp_backend()

    m = factor * I.shape[0] / 2
    n = factor * I.shape[1] / 2

    N = xp.min(xp.array([m, n]))
    N_val = int(N)
    I = I[int(I.shape[0] // 2 - N_val): int(I.shape[0] // 2 + N_val),
                int(I.shape[1] // 2 - N_val): int(I.shape[1] // 2 + N_val)]

    return I

# Spatial Windows:
def apply_raised_cosine_window(image):
    xp = xp_backend()
    rows, cols = image.shape
    i = xp.arange(rows)
    j = xp.arange(cols)
    window = 0.5 * (1 + xp.cos(xp.pi * (2 * i[:, None] - rows) / rows)) * \
             0.5 * (1 + xp.cos(xp.pi * (2 * j - cols) / cols))
    return image * window

def blackman_harris_window(size: int, a0: float, a1: float, a2: float, a3: float):
    xp = xp_backend()
    n = xp.arange(size)
    window = (a0
              - a1 * xp.cos(2 * xp.pi * n / (size - 1))
              + a2 * xp.cos(4 * xp.pi * n / (size - 1))
              - a3 * xp.cos(6 * xp.pi * n / (size - 1)))
    return window

def apply_blackman_harris_window(image,
                                 a0: float = 0.35875, a1: float = 0.48829,
                                 a2: float = 0.14128, a3: float = 0.01168):
    xp = xp_backend()
    height, width = image.shape
    window_row = blackman_harris_window(width, a0, a1, a2, a3)
    window_col = blackman_harris_window(height, a0, a1, a2, a3)
    image_windowed = xp.outer(window_col, window_row) * image
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