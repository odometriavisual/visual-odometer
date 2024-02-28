from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import os

from virtualencoder.visualodometry.debugging_tools import debugging_estimate_shift, generate_plot_variables, have_number_out_range
from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
from virtualencoder.visualodometry.utils import get_imgs


# Função have_number_out_range modificada
def any_phase_out(variables, deltax, deltay):
    phasey_unwrapped = variables["phasey_unwrapped"]
    phasex_unwrapped = variables["phasex_unwrapped"]
    center = phasey_unwrapped.size//2
    if have_number_out_range(phasey_unwrapped):
        if deltay > 0.3:
            return True
    if have_number_out_range(phasex_unwrapped):
        if deltax > 0.3:
            return True
    return False

img_counter = 0

print("Tentando pegar a lista de imagens")
imgs = get_imgs(n=None,data_root="data/DATASETS-21.02.2024/-9/LONG/2024-02-21_11-13-59")
print("Lista de imagens Ok, iniciando geração do vídeo")


# Caminho onde os plots serão salvos como imagens
output_folder = "plots/"

# Certifique-se de que a pasta de saída exista, caso contrário, crie-a
os.makedirs(output_folder, exist_ok=True)

# Lista para armazenar os caminhos dos arquivos salvos
plot_paths = []
total_deltax = 0
total_deltay = 0

# Loop sobre os pares de imagens e gerar e salvar os plots
# Loop sobre os pares de imagens e gerar e salvar os plots
img_1 = None
img_0 = None
for j in range(1):
    for i in tqdm(range(0, len(imgs) - 1, 1)):  # Itera sobre índices ímpares
        img_0 = imgs[i]
        if img_1 is None:
            img_1 = img_0

            continue
        img_0 = apply_border_windowing_on_image(img_0, border_windowing_method=None)
        img_1 = apply_border_windowing_on_image(img_1, border_windowing_method=None)

        deltay, deltax, variables = debugging_estimate_shift(img_0, img_1, finge_filter=True, finge_filter_threshold=0.03)
        additional_info = {
            "img_0_id": i,
            "img_1_id": i + 1,
        }
        variables.update(additional_info)
        total_deltax += deltax
        total_deltay += deltay

        #if any_phase_out(variables, deltax, deltay) or True:  # Verifica se deve salvar a imagem
        plot_path = os.path.join(output_folder, f"plot_{i}.png")
        text_title = f"deltax: {deltax:.2f}, deltay: {deltay:.2f}\n total_x: {total_deltax:.2f}, total_y:{total_deltay:.2f}"
        plt = generate_plot_variables(img_0, img_1, variables, text_title)
        plt.savefig(plot_path)
        plt.close()
        plot_paths.append(plot_path)
        img_1 = img_0 #compare_two_variables_genimgs

# Configurações para o vídeo
fps = 10  # Quadros por segundo
frame_size = (800, 600)  # Tamanho do frame

# Crie o objeto de vídeo
video_writer = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, frame_size)

# Leia as imagens salvas e adicione ao vídeo
for plot_path in plot_paths:
    img = cv2.imread(plot_path)
    img = cv2.resize(img, frame_size)
    video_writer.write(img)

# Libere o objeto de vídeo
video_writer.release()

print("Vídeo gerado com sucesso!")
