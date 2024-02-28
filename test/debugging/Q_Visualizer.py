import numpy as np
from matplotlib import pyplot as plt

from virtualencoder.visualodometry.dsp_utils import phase_fringe_filter
from virtualencoder.visualodometry.image_utils import apply_border_windowing_on_image
from virtualencoder.visualodometry.debugging_tools import debugging_estimate_shift, generate_plot_variables
from virtualencoder.visualodometry.utils import get_imgs

print("Tentando pegar a lista de imagens")
imgs = get_imgs(n=None,data_root="data/DATASETS-21.02.2024/-9/LONG/2024-02-21_11-13-59")
print("Lista de imagens Ok, iniciando procura de erros")

img_0 = imgs[75]
img_1 = imgs[77]
img_0 = apply_border_windowing_on_image(img_0)
img_1 = apply_border_windowing_on_image(img_1)

deltay, deltax, variables = debugging_estimate_shift(img_0,img_1)
Q = variables["Q"]


print(deltax, deltay)

plt.figure()
plt.imshow(np.imag(Q), cmap='gray')
plt.title('Magnitude da Matriz Q')
plt.colorbar()
plt.show()

deltay, deltax, variables = debugging_estimate_shift(img_0,img_1,finge_filter=True, finge_filter_threshold=0.03)
Q = variables["Q"]

plt.figure()
plt.imshow(np.imag(Q), cmap='gray')
plt.title('Magnitude da Matriz Q')
plt.colorbar()
plt.show()

