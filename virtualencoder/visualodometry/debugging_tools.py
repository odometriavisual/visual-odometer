import numpy as np
from numpy.random import RandomState
from scipy.sparse.linalg import svds
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

from virtualencoder.visualodometry.dsp_utils import image_preprocessing, normalize_product, phase_fringe_filter
from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
from virtualencoder.visualodometry.svd_decomposition import phase_unwrapping, linear_regression

def debugging_svd_estimate_shift(phase_vec, N):
    # É uma versão que retorma mais informações do processo apenas para debugging
    # Phase unwrapping:
    phase_unwrapped = phase_unwrapping(phase_vec)
    r = np.arange(0, phase_unwrapped.size)
    M = r.size // 2
    x = r               #  [M-80:M-20]
    y = phase_unwrapped #[M-80:M-20]
    mu, c = linear_regression(x, y)
    delta = mu * N / (2 * np.pi)
    return -delta, phase_unwrapped, mu, c



def debugging_estimate_shift(img_0, img_1, method='Stone_et_al_2001', random_state=None, finge_filter = False, finge_filter_threshold = 0.03):
    M, N = img_0.shape
    img_0_processed = image_preprocessing(img_0, method)
    img_1_processed = image_preprocessing(img_1, method)

    Q = normalize_product(img_0_processed, img_1_processed)

    if finge_filter is True:
        print(Q)
        Q = phase_fringe_filter(Q, threshold=finge_filter_threshold)

    qu, s, qv = svds(Q, k=1, random_state=random_state)
    ang_qu = np.angle(qu[:, 0])
    ang_qv = np.angle(qv[0, :])
    # Deslocamento no eixo x é equivalente a deslocamento ao longo do eixo das colunas e eixo y das linhas:
    deltay, phasey_unwrapped, muy, cy = debugging_svd_estimate_shift(ang_qu, M)
    deltax, phasex_unwrapped, mux, cx = debugging_svd_estimate_shift(ang_qv, N)

    variables = {
        "img_0_processed": img_0_processed,
        "img_1_processed": img_1_processed,
        "Q": Q,
        "qu": qu,
        "s": s,
        "qv": qv,
        "ang_qu": ang_qu,
        "ang_qv": ang_qv,
        "phasex_unwrapped": phasex_unwrapped,
        "phasey_unwrapped": phasey_unwrapped,
        "muy": muy,
        "mux": mux,
        "cy": cy,
        "cx": cx,
        "random_state": random_state,
        "img_0_id": 0,
        "img_1_id": 1
    }
    return deltax, deltay, variables


def generate_plot_variables(img_0, img_1, variables, center_text=""):
    """
    gera um plot baseado nas variáveis da função debugging_estimate_shift
    exemplo de uso:

        deltax, deltay, variables = debugging_estimate_shift(img_0, img_1)
        plt = generatePlotDebuggingImages(img_0, img_1, variables, "Qualquer Texto")
        plt.show()
    """

    xx_values = np.linspace(0, 300, 1000)  # Valores arbitrários para x
    xy_values = variables["mux"] * xx_values + variables["phasex_unwrapped"][0]

    yx_values = np.linspace(0, 300, 1000)  # Valores arbitrários para x
    yy_values = variables["muy"] * yx_values + variables["phasey_unwrapped"][0]

    fig, ax = plt.subplots(2, 5, figsize=(12, 8))  # 3 linhas, 3 colunas

    ax[0, 0].imshow(img_0, cmap='gray')
    ax[0, 0].set_title(f'{variables["img_0_id"]}')

    ax[1, 0].imshow(img_1, cmap='gray')
    ax[1, 0].set_title(f'{variables["img_1_id"]}')

    ax[0, 1].imshow(np.log10(abs(variables["img_0_processed"])), cmap='gray')
    ax[0, 1].set_title(f'FFT 0')

    ax[1, 1].imshow(np.log10(abs(variables["img_1_processed"])), cmap='gray')
    ax[1, 1].set_title(f'FFT 1')

    x = range(0, variables["qu"].size)
    ax[0, 2].plot(variables["qu"])
    ax[0, 2].scatter(x, variables["qu"], s=5, color="red")
    ax[0, 2].set_title(f'Angulo qu')

    x = range(0, variables["qv"].size)
    ax[1, 2].scatter(x, variables["qv"], s=5, color="red")
    ax[1, 2].plot(variables["qv"])
    ax[1, 2].set_title(f'Angulo qv')

    x = range(0, variables["ang_qu"].size)
    ax[0, 3].scatter(x, variables["ang_qu"], s=5, color="red")
    ax[0, 3].plot(variables["ang_qu"])
    ax[0, 3].set_title(f'Angulo ang_qu')

    x = range(0, variables["ang_qv"].size)
    ax[1, 3].scatter(x, variables["ang_qv"], s=5, color="red")
    ax[1, 3].plot(variables["ang_qv"])
    ax[1, 3].set_title(f'Angulo ang_qv')

    ax[0, 4].plot(variables["phasey_unwrapped"])
    ax[0, 4].plot(yx_values, yy_values)
    ax[0, 4].set_title(f'Angulo Y')

    ax[1, 4].plot(variables["phasex_unwrapped"])
    ax[1, 4].plot(xx_values, xy_values)
    ax[1, 4].set_title(f'Angulo X')

    text_obj = fig.text(0.5, 0.94, center_text, ha='center', va='center', fontsize=12, transform=fig.transFigure)

    return plt


def compare_two_variables_genimgs(img_0, img_1, variables_1, variables_2, center_text=""):
    """
    gera um plot baseado nas variáveis da função debugging_estimate_shift
    exemplo de uso:

        deltax, deltay, variables = debugging_estimate_shift(img_0, img_1)
        plt = generatePlotDebuggingImages(img_0, img_1, variables, "Qualquer Texto")
        plt.show()
    """

    xx_values = np.linspace(0, 300, 1000)  # Valores arbitrários para x
    xy_values = variables_1["muy"] * xx_values + variables_1["phasex_unwrapped"][0]

    yx_values = np.linspace(0, 300, 1000)  # Valores arbitrários para x
    yy_values = variables_2["muy"] * yx_values + variables_2["phasex_unwrapped"][0]

    fig, ax = plt.subplots(2, 5, figsize=(12, 8))  # 3 linhas, 3 colunas

    ax[0, 0].imshow(img_0, cmap='gray')
    ax[0, 0].set_title(f'{variables_1["img_0_id"]}')

    ax[1, 0].imshow(img_1, cmap='gray')
    ax[1, 0].set_title(f'{variables_1["img_1_id"]}')

    ax[0, 1].imshow(np.log10(abs(variables_1["img_0_processed"])), cmap='gray')
    ax[0, 1].set_title(f'FFT 0')

    ax[1, 1].imshow(np.log10(abs(variables_2["img_1_processed"])), cmap='gray')
    ax[1, 1].set_title(f'FFT 1')

    x = range(0, variables_1["qu"].size)
    ax[0, 2].plot(variables_1["qu"])
    ax[0, 2].scatter(x, variables_1["qu"], s=5, color="red")
    ax[0, 2].set_title(f'Angulo qu')

    x = range(0, variables_2["qu"].size)
    ax[1, 2].scatter(x, variables_2["qu"], s=5, color="red")
    ax[1, 2].plot(variables_2["qu"])
    ax[1, 2].set_title(f'Angulo qu')

    x = range(0, variables_1["ang_qu"].size)
    ax[0, 3].scatter(x, variables_1["ang_qu"], s=5, color="red")
    ax[0, 3].plot(variables_1["ang_qu"])
    ax[0, 3].set_title(f'Angulo ang_qu')

    x = range(0, variables_2["ang_qu"].size)
    ax[1, 3].scatter(x, variables_2["ang_qu"], s=5, color="red")
    ax[1, 3].plot(variables_2["ang_qu"])
    ax[1, 3].set_title(f'Angulo ang_qu')

    ax[0, 4].plot(variables_1["phasey_unwrapped"])
    #ax[0, 4].plot(yx_values, yy_values)
    ax[0, 4].set_title(f'Angulo Y')

    ax[1, 4].plot(variables_2["phasey_unwrapped"])
    #ax[1, 4].plot(xx_values, xy_values)
    ax[1, 4].set_title(f'Angulo Y')

    text_obj = fig.text(0.5, 0.94, center_text, ha='center', va='center', fontsize=12, transform=fig.transFigure)

    return plt


def error_finder(img_0, img_1, estimated_value_y=None, estimated_value_x=None, max_variance_x=0.03, max_variance_y=0.03,
                 counter_limit=100, counter_only=False, random_mode=True, method='Stone_et_al_2001', borderWindowing=False):

    if estimated_value_x is None:
        estimated_value_x, estimated_value_y, _initial_variables = debugging_estimate_shift(img_0, img_1, random_state=0)

    counter = 0
    deltax = 0
    deltay = 0
    variables = {}
    error_counter = np.zeros(2)
    absolute_varianxe_x = abs(max_variance_x * estimated_value_x)
    absolute_variance_y = abs(max_variance_y * estimated_value_y)
    while counter_limit > counter:
        counter = counter + 1

        if random_mode is True:
            deltax, deltay, variables = debugging_estimate_shift(img_0, img_1, method)
        else:
            deltax, deltay, variables = debugging_estimate_shift(img_0, img_1, method, counter)

        if (deltay > estimated_value_y + absolute_variance_y) or (deltay < estimated_value_y - absolute_variance_y):
            error_counter[0] = error_counter[0] + 1
            if counter_only is False:
                return deltax, deltay, variables, True

        if (deltax > estimated_value_x + absolute_varianxe_x) or (deltax < estimated_value_x - absolute_varianxe_x):
            error_counter[1] = error_counter[1] + 1
            if counter_only is False:
                return deltax, deltay, variables, True

    return deltax, deltay, variables, False

def have_number_out_range(arr):
    threshold = (np.max(arr) - np.min(arr)) / 2
    n = len(arr)
    if n < 3:
        return False

    for i in range(1, n - 1):
        diff_prev = abs(abs(arr[i]) - abs(arr[i - 1]))
        diff_next = abs(abs(arr[i]) - abs(arr[i + 1]))
        if diff_prev >= threshold or diff_next >= threshold:
            return True

    return False



