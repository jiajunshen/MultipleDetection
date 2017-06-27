import h5py
import numpy as np
import plot3D
from voxelgrid import VoxelGrid

hf = h5py.File("/hdd/Documents/Data/3D-MNIST/full_dataset_vectors.h5", "r")
img = np.array([hf["X_train"][i].reshape(16, 16, 16) for i in range(100)])
#img = plot3D.array_to_color(img)
"""

from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

l = plot3d(x_list, y_list, z_list)

l.show()
"""

#plot3D.plot_points(img, size=0.01)

from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define several atoms, these are numpy arrays of length 3
# randomly pulled from a uniform distribution between -1 and 1
"""
x_list = []
y_list = []
z_list = []
dx, dy, dz = np.meshgrid(np.linspace(0, 15, 16), np.linspace(0, 15, 16), np.linspace(0, 15, 16))
dx = np.array(dx, dtype=np.int32).flatten()
dy = np.array(dy, dtype=np.int32).flatten()
dz = np.array(dz, dtype=np.int32).flatten()
for i in range(4096):
    if img[dx[i], dy[i], dz[i]] > 0.5:
        x_list.append(dx[i])
        y_list.append(dy[i])
        z_list.append(dz[i])

# Define a list of colors to plot atoms

colors = ['r']


ax = plt.subplot(111, projection='3d')

# Plot scatter of points
ax.scatter3D(x_list, y_list, z_list, c=colors)
"""
with h5py.File("/hdd/Documents/Data/3D-MNIST/train_point_clouds.h5", "r") as hf1:

    a = hf1["0"]
    b = hf1["1"]

    digit_a = (a["img"][:], a["points"][:], a.attrs["label"])
    digit_b = (b["img"][:], b["points"][:], b.attrs["label"])

plt.subplot(121)
plt.title("DIGIT A: " + str(digit_a[2]))
plt.imshow(digit_a[0])

plt.subplot(122)
plt.title("DIGIT B: " + str(digit_b[2]))
plt.imshow(digit_b[0])

a_voxelgrid = VoxelGrid(digit_a[1], x_y_z=[8, 8, 8])
b_voxelgrid = VoxelGrid(digit_b[1], x_y_z=[8, 8, 8])

plot3D.plot_voxelgrid(a_voxelgrid)
img1 = plot3D.array_to_color(a_voxelgrid.vector.flatten())
plot3D.plot_points(img1)

#img = np.load("/hdd/Documents/Data/ModelNet10/X_train_example.npy")
#img_label = np.load("/hdd/Documents/Data/ModelNet10/Y_train_example.npy")
img = np.load("/hdd/Documents/Data/3D-MNIST/X_train.npy")
img_label = np.load("/hdd/Documents/Data/3D-MNIST/Y_train.npy")

size = 40
image_all = np.zeros((size * 5, size * 5, size))
for i in range(5):
    currentImage = img[img_label == i][:5]
    for j in range(5):
        image_all[i*size:i*size + 40, j*size:j*size+40] = currentImage[j, ::-1, ::-1, ::-1]
plot3D.plot_voxelgridnew(image_all)
