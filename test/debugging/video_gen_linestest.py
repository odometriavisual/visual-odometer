import cv2
from virtualencoder.visualodometry.dsp_utils import image_preprocessing
from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
from virtualencoder.visualodometry.svd_decomposition import optimized_svd_method
from virtualencoder.visualodometry.utils import get_imgs, get_rgb_imgs
import numpy as np

# Configurações de vídeo
fps = 30.0

# Configurando variáveis para posicionamento de texto
y1 = 30
y2 = 70

print("Pegando a lista de imagens")
imgs = get_imgs(n=None, data_root="data/DATASETS-21.02.2024/-10/LONG/2024-02-21_11-08-31")
rgb_imgs = get_rgb_imgs(n=None, data_root="data/DATASETS-21.02.2024/-10/LONG/2024-02-21_11-08-31")

print("Lista de imagens OK")

M = None
N = None
total_deltax = 0
total_deltay = 0
all_lines_coordinates = []  # Lista para armazenar todas as coordenadas das linhas desenhadas

# Inicializa o gravador de vídeo com o codec MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


phase_windowing = None
border_windowing_method = "raised_cosine"


output_video_path = 'output/'
text_description = ""
frame_divider = 1
if frame_divider == 1:
    text_frame_removing = "Todos os frames"
else:
    text_frame_removing = f"Apenas 1/{frame_divider} das imagens"

if phase_windowing == None:
    text_description += "Sem janelamento de fase"
    output_video_path += 'no_phase_windowing-'

elif phase_windowing == "central":
    text_description += "Com janelamento central de fase"
    output_video_path += 'central_phase_windowing-'

if border_windowing_method == None:
    text_description += " e sem o escurecimento de bordas"
    output_video_path += 'no_border_black_windowing'
elif border_windowing_method == "blackman_harris":
    output_video_path += 'with_blackman_harris_window'
    text_description += " e com escurecimento de bordas (blackman harris)"
elif border_windowing_method == "raised_cosine":
    output_video_path += 'with_raised_cosine_window'
    text_description += ",com escurecimento de bordas (raised cosine)"

text_description += " e o finge filter"

if frame_divider != 1:
    output_video_path += f"_frame_divider_by_{frame_divider}"



output_video_path += '.mp4'

print(output_video_path)
print(text_description)

out = cv2.VideoWriter(output_video_path, fourcc, fps,
                      (imgs[0].shape[1], imgs[0].shape[0]))  # Usa a resolução das imagens originais

print("Iniciando processamento")
for i, img in enumerate(imgs):
    if M is None:
        M, N = img.shape
        center_x = rgb_imgs[i].shape[1] // 2
        center_y = rgb_imgs[i].shape[0] // 2
        y3 = rgb_imgs[i].shape[0] - 10
        y4 = y3 - 20
        sumdeltax = 0
        sumdeltay = 0
        windowed_img = apply_border_windowing_on_image(img, border_windowing_method=border_windowing_method)
        img_processed_old = image_preprocessing(windowed_img)
    elif i%frame_divider == 1 or frame_divider == 1:
        windowed_img = apply_border_windowing_on_image(img, border_windowing_method=border_windowing_method)
        img_processed = image_preprocessing(windowed_img)
        deltax, deltay = optimized_svd_method(img_processed, img_processed_old, M, N, phase_windowing=phase_windowing)
        total_deltax += deltax
        total_deltay += deltay

        all_lines_coordinates.insert(0, [deltax, deltay])
        img_with_lines = rgb_imgs[i].copy()  # Faz uma cópia para desenhar as linhas

        soma_deltax = 0
        soma_deltay = 0
        j = 0
        for coordenada in all_lines_coordinates:
            j = j + 1
            color = (0,255,0)
            if j == 1:
                color = (0,255, 255)
            cv2.line(img_with_lines,
                     (int(center_x+soma_deltax), int(center_y+soma_deltay)),
                     (int(coordenada[0]+center_x+soma_deltax), int(coordenada[1]+center_y+soma_deltay)),
                     color, 2)
            soma_deltax += coordenada[0]
            soma_deltay += coordenada[1]


        cv2.putText(img_with_lines, f'Total DeltaX: {total_deltax:.2f}', (10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1)
        cv2.putText(img_with_lines, f'Total DeltaY: {total_deltay:.2f}', (10, y2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1)
        cv2.putText(img_with_lines, text_description, (10, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
        cv2.putText(img_with_lines, text_frame_removing, (10, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

        # Exibir o frame para depuração
        cv2.imshow('Frame', img_with_lines)
        cv2.waitKey(1)  # Aguarda um curto período para exibir o frame

        # Adicionar o frame ao vídeo
        out.write(img_with_lines)
        img_processed_old = img_processed


num_repeticoes = 30
img_0 = rgb_imgs[0].copy()
img_1 = rgb_imgs[-1].copy()

img_0_gray_processed = image_preprocessing(imgs[1])
img_1_gray_processed = image_preprocessing(imgs[-1])

deltax_esperado, deltay_esperado = optimized_svd_method(img_0_gray_processed, img_1_gray_processed, M, N)

for _ in range(num_repeticoes):
    cv2.putText(img_0, "Imagem inicial", (10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 1)
    out.write(img_0)
for _ in range(num_repeticoes):
    cv2.putText(img_1, "Imagem final", (10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 1)
    out.write(img_1)
img_2 = rgb_imgs[-1].copy()
# for _ in range(num_repeticoes):
#     msg1 = f"deltax esperado {deltax_esperado:.2f}. total deltax: {total_deltax:.2f}"
#     msg2 = f"deltay esperado {deltay_esperado:.2f}. total deltay: {total_deltay:.2f}"
#     cv2.putText(img_2, msg1, (10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                 (255, 255, 255), 1)
#     cv2.putText(img_2, msg2, (10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                 (255, 255, 255), 1)
#     cv2.putText(img_2, text_description, (10, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                 (255, 255, 255), 1)
#     out.write(img_2)


# Libera o gravador de vídeo
out.release()
cv2.destroyAllWindows()  # Fecha a janela de visualização após a conclusão
print("Vídeo salvo com sucesso!")
