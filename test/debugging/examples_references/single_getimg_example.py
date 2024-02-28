from virtualencoder.visualodometry.debugging_tools import debugging_estimate_shift, generate_plot_variables
from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
from virtualencoder.visualodometry.utils import get_imgs

print("Tentando pegar a lista de imagens")
imgs = get_imgs(n=None,data_root="data/DATASETS-21.02.2024/-9/LONG/2024-02-21_11-13-59")
print("Lista de imagens Ok, iniciando procura de erros")

img_0 = apply_border_windowing_on_image(imgs[0], border_windowing_method="raised_cosine")
img_1 = apply_border_windowing_on_image(imgs[-1], border_windowing_method="raised_cosine")
deltay, deltax, variables = debugging_estimate_shift(img_0,img_1)
text_title = f"deltax: {deltax:.2f}, deltay: {deltay:.2f}"
plt = generate_plot_variables(img_0, img_1, variables, text_title)
plt.show()