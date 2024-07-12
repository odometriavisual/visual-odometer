# utils.py

import numpy as np

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
