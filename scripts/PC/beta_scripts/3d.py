
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from stl import mesh

# Carregar o arquivo STL
stl_filename = ('/home/henriqueguerra/Downloads/500acq.stl')
mesh = mesh.Mesh.from_file(stl_filename)

# Configurar a figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Adicionar o modelo STL à figura
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))

# Definir limites do gráfico
scale = mesh.points.flatten()
ax.auto_scale_xyz(scale, scale, scale)

# Exibir a figura
plt.show()
