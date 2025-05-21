import numpy as np
import cv2

width, height = 640, 480
img1 = np.zeros((height, width), dtype=np.uint8)
img2 = np.zeros_like(img1)

# Parâmetros dos círculos (posição e raio)
circles = [
    (100, 100, 30),
    (300, 200, 50),
    (500, 400, 40),
]

# Deslocamento da segunda imagem (em pixels)
dx, dy = 15, 10

# Desenhar círculos na primeira imagem
for (x, y, r) in circles:
    cv2.circle(img1, (x, y), r, 255, thickness=-1)  # branco preenchido

# Desenhar círculos deslocados na segunda imagem
for (x, y, r) in circles:
    new_x = x + dx
    new_y = y + dy
    if 0 <= new_x < width and 0 <= new_y < height:
        cv2.circle(img2, (new_x, new_y), r, 255, thickness=-1)

# Salvar imagens
cv2.imwrite("img.png", img1)
cv2.imwrite("img_translated.png", img2)

# Exibir (opcional)
cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
