from virtualencoder.visualodometry.debugging_tools import debugging_estimate_shift, generate_plot_variables
from virtualencoder.visualodometry.image_utils import img_gen

img_0 = img_gen(0,0)
img_1 = img_gen(10,100)
deltay, deltax, variables = debugging_estimate_shift(img_0,img_1)
print(deltax, deltay)
text_title = f"deltax: {deltax:.2f}, deltay: {deltay:.2f}"
plt = generate_plot_variables(img_0, img_1, variables, text_title)
plt.show()