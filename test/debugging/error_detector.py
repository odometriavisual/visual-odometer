from virtualencoder.visualodometry.debugging_tools import error_finder
from virtualencoder.visualodometry.utils import get_imgs

img_counter = 0

print("Tentando pegar a lista de imagens")
imgs = get_imgs(n=None,data_root="data/DATASETS-21.02.2024/-9/LONG/2024-02-21_11-13-59")
print("Lista de imagens Ok, iniciando procura de erros")

for img_0, img_1 in zip(imgs, imgs[1:]):
    img_counter += 1
    print(f"Iniando procura na imagem {img_counter} e {img_counter+1}")
    deltax, deltay, variables, error = error_finder(img_0, img_1)
    if error is True:
        print("Error found")
        print(img_counter)
        print(variables["random_state"])
    else:
        print("no error found")