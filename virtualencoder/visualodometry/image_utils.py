import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter


# Mantendo os métodos repetidos para compatbilidade, será removido nas próximas versões
# Utilities associated with digital image processing.

def generate_window(img_dim, window_type=None):
    if window_type is None:
        return np.ones(img_dim)
    elif window_type == 'Blackman-Harris':
        window1dy = signal.windows.blackmanharris(img_dim[0])
        window1dx = signal.windows.blackmanharris(img_dim[1])
        window2d = np.sqrt(np.outer(window1dy, window1dx))
        return window2d
    elif window_type == 'Blackman':
        window1dy = np.abs(np.blackman(img_dim[0]))
        window1dx = np.abs(np.blackman(img_dim[1]))
        window2d = np.sqrt(np.outer(window1dy, window1dx))
        return window2d


def apply_window(img1, img2, window_type):
    window = generate_window(img1.shape, window_type)
    return img1 * window, img2 * window


def autocontrast(img):
    autocontrast_img = img
    input_minval = autocontrast_img.min()
    input_maxval = autocontrast_img.max()
    output_img = (autocontrast_img - input_minval) / (input_maxval - input_minval) * 255
    return output_img


def gaussian_noise(image, snr):
    signal_energy = np.power(np.linalg.norm(image), 2)
    # snr = 20 * log10(signal_energy / noise_energy)
    # Therefore: noise_energy = 10^(- snr/20 + log10(signal_energy)
    noise_energy = np.power(10, np.log10(signal_energy) - snr / 20)

    # noise_energy ^2 =~ M * sigma^2 => sigma = sqrt(noise_energy^2 / M),
    # where sigma is the standard deviation and M is the signal length
    M = image.size  # Total number of elements
    sd = np.sqrt(np.power(noise_energy, 2) / M)
    sd = np.mean(image) * 0.2
    noise_signal = np.random.normal(0, sd, M)
    flattened_image = image.flatten(order="F") + noise_signal
    noisy_image = np.reshape(flattened_image, image.shape, order='F')
    return noisy_image


def salt_and_pepper(image, prob=0.05):
    # If the specified `prob` is negative or zero, we don't need to do anything.
    if prob <= 0:
        return image

    arr = np.asarray(image)
    original_dtype = arr.dtype

    # Derive the number of intensity levels from the array datatype.
    intensity_levels = 2 ** 8

    min_intensity = 0
    max_intensity = intensity_levels - 1

    # Generate an array with the same shape as the image's:
    # Each entry will have:
    # 1 with probability: 1 - prob
    # 0 or np.nan (50% each) with probability: prob
    random_image_arr = np.random.choice(
        [min_intensity, 1, np.nan], p=[prob / 2, 1 - prob, prob / 2], size=arr.shape
    )

    # This results in an image array with the following properties:
    # - With probability 1 - prob: the pixel KEEPS ITS VALUE (it was multiplied by 1)
    # - With probability prob/2: the pixel has value zero (it was multiplied by 0)
    # - With probability prob/2: the pixel has value np.nan (it was multiplied by np.nan)
    # We need to to `arr.astype(np.float)` to make sure np.nan is a valid value.
    salt_and_peppered_arr = arr.astype(np.float) * random_image_arr

    # Since we want SALT instead of NaN, we replace it.
    # We cast the array back to its original dtype so we can pass it to PIL.
    salt_and_peppered_arr = np.nan_to_num(
        salt_and_peppered_arr, nan=max_intensity
    ).astype(original_dtype)

    return salt_and_peppered_arr


def apply_blackman_harris_window(image):
    height, width = image.shape
    window_row = blackman_harris_window(width)
    window_col = blackman_harris_window(height)
    image_windowed = np.outer(window_col, window_row) * image
    return image_windowed


def blackman_harris_window(size, a0=0.35875, a1=0.48829, a2=0.14128, a3=0.01168):
    # a0, a1, a2 e a3 são os coeficientes de janelamento
    # Criação do vetor de amostras
    n = np.arange(size)
    # Cálculo da janela de Blackman-Harris
    window = a0 - a1 * np.cos(2 * np.pi * n / (size - 1)) + a2 * np.cos(4 * np.pi * n / (size - 1)) - a3 * np.cos(
        6 * np.pi * n / (size - 1))
    return window


def img_gen(dx=0, dy=0, width=512, height=600, zoom=1, angle=0):
    img = np.zeros((width, height), dtype=float)
    prng = RandomState(1234)
    circ = prng.rand(100, 4)

    # Definir a grade de coordenadas X e Y
    X = np.arange(width)
    Y = np.arange(height)

    # Aplicar o deslocamento
    X = X - (width) + dx
    Y = Y - (height) + dy

    # Aplicar o zoom
    X = (X / zoom) + (width / 2)
    Y = (Y / zoom) + (height / 2)

    X, Y = np.meshgrid(X, Y)

    for i in range(circ.shape[0]):
        # Calcular o ângulo de rotação para a máscara
        angle_rad = np.radians(angle)

        # Calcular as coordenadas X e Y ajustadas com rotação para cada ponto da malha
        X_rotated = X * np.cos(angle_rad) - Y * np.sin(angle_rad)
        Y_rotated = X * np.sin(angle_rad) + Y * np.cos(angle_rad)

        # Ajustar as coordenadas dos centros dos círculos para centralizar
        circle_center_x = circ[i, 0] * width - (width / 2)
        circle_center_y = circ[i, 1] * height - (height / 2)

        # Calcular a máscara com base nas coordenadas rotacionadas e ajustadas
        mask = ((X_rotated - circle_center_x) ** 2 + (Y_rotated - circle_center_y) ** 2) < (circ[i, 2] * 100) ** 2

        # Atribuir à imagem
        img[mask.T] = circ[i, 3]  # Transpor a máscara antes de atribuir à imagem

    return img


def add_gaussian_noise(img, noise_level=0.1):
    """
    Adiciona ruído gaussiano à imagem.

    Args:
        img (ndarray): Imagem de entrada.
        noise_level (float): Nível de ruído, padrão é 0.1.

    Returns:
        ndarray: Imagem com ruído gaussiano adicionado.
    """
    # Calcula a média e o desvio padrão da imagem
    mean = np.mean(img)
    std_dev = np.std(img)

    # Calcula o ruído gaussiano
    noise = np.random.normal(mean, std_dev * noise_level, img.shape)

    # Adiciona o ruído à imagem
    noisy_img = img + noise

    # Garante que os valores da imagem resultante estejam no intervalo [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255)

    return noisy_img.astype(np.uint8)  # Converte para o tipo de dados uint8


def add_blur(img, kernel_size=3):
    blurred_img = uniform_filter(img.astype(float), size=kernel_size)
    return blurred_img  # Converte para o tipo de dados uint8


def add_noise(img, noise_type='gaussian', noise_level=0.1):
    if noise_type == 'gaussian':
        noisy_img = add_gaussian_noise(img, noise_level)
    else:
        raise ValueError("Tipo de ruído não suportado. Escolha entre 'gaussian', 'salt_and_pepper' ou 'speckle'.")
    return noisy_img


def filter_array_by_maxmin(arr):
    # Filtro para remover valores absolutos muito diferentes dentro de um array
    filtered_arr = []
    threshold = (arr.max() - arr.min()) / 4
    n = len(arr)

    # Se o array tiver menos que 3 elementos, não há o que filtrar
    if n < 3:
        return arr

    # Adiciona o primeiro elemento
    filtered_arr.append(arr[0])

    # Itera sobre os elementos do array, ignorando o primeiro e o último elemento
    for i in range(1, n - 1):
        diff_prev = abs(abs(arr[i]) - abs(arr[i - 1]))
        diff_next = abs(abs(arr[i]) - abs(arr[i + 1]))

        # Verifica se a diferença entre o elemento atual e o anterior, e entre o atual e o próximo
        # são menores que o threshol
        if (diff_prev < threshold and diff_next < threshold):
            filtered_arr.append(arr[i])
        else:
            median = arr[i - 1] / 2 + arr[i + 1] / 2
            if abs(median - arr[i - 1]) < threshold:
                filtered_arr.append(median)

    # Adiciona o último elemento
    filtered_arr.append(arr[-1])
    return np.array(filtered_arr)


def apply_raised_cosine_window(image):
    rows, cols = image.shape
    i = np.arange(rows)
    j = np.arange(cols)
    window = 0.5 * (1 + np.cos(np.pi * (2 * i[:, None] - rows) / rows)) * \
             0.5 * (1 + np.cos(np.pi * (2 * j - cols) / cols))
    return image * window


def apply_border_windowing_on_image(image, border_windowing_method="blackman_harris"):
    if border_windowing_method == "blackman_harris":
        return apply_blackman_harris_window(image)
    elif border_windowing_method == "raised_cosine":
        return apply_raised_cosine_window(image)
    elif border_windowing_method == None:
        return image
