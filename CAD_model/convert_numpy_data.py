import numpy as np
from os import listdir
from os.path import isfile, join
plain_file_path = "/hdd/Documents/Data/ShapeNetCoreV2/plain_image_4"
plain_files = [join(plain_file_path, f) for f in listdir(plain_file_path) if isfile(join(plain_file_path, f))]
texture_file_path = '/hdd/Documents/Data/ShapeNetCoreV2/texture_image_4'
texture_files = [join(texture_file_path, f) for f in listdir(texture_file_path) if isfile(join(texture_file_path, f))]

# Source => Target = (BGColor + Source) =
# Target.R = ((1 - Source.A) * BGColor.R) + (Source.A * Source.R)
# Target.G = ((1 - Source.A) * BGColor.G) + (Source.A * Source.G)
# Target.B = ((1 - Source.A) * BGColor.B) + (Source.A * Source.B)

from scipy import misc
plain_images = []
for f in plain_files:
    image = misc.imread(f)
    mask = np.array(np.product(image, axis = 2) == 237 * 237 * 255 * 255).reshape(500,500,1)
    mask_all = np.repeat(mask, 3, axis = 2)
    
    image = image / 255.0
    bgcolor = np.array([237, 237, 255]).reshape(1, 1, 3) / 255.0
    new_image = np.zeros((500, 500, 3))
    new_image[:,:,0] = (1 - image[:,:,3]) * image[:,:,0] + image[:,:,3] * image[:,:,0]
    new_image[:,:,1] = (1 - image[:,:,3]) * image[:,:,1] + image[:,:,3] * image[:,:,1]
    new_image[:,:,2] = (1 - image[:,:,3]) * image[:,:,2] + image[:,:,3] * image[:,:,2]
    new_image = bgcolor * mask_all + new_image * (1 - mask_all)
    plain_images.append(new_image)
plain_images = np.asarray(plain_images)

texture_images = []
for f in texture_files:
    image = misc.imread(f)
    mask = np.array(np.product(image, axis = 2) == 237 * 237 * 255 * 255).reshape(500,500,1)
    mask_all = np.repeat(mask, 3, axis = 2)
    
    image = image / 255.0
    bgcolor = np.array([237, 237, 255]).reshape(1, 1, 3) / 255.0
    new_image = np.zeros((500, 500, 3))
    new_image[:,:,0] = (1 - image[:,:,3]) * image[:,:,0] + image[:,:,3] * image[:,:,0]
    new_image[:,:,1] = (1 - image[:,:,3]) * image[:,:,1] + image[:,:,3] * image[:,:,1]
    new_image[:,:,2] = (1 - image[:,:,3]) * image[:,:,2] + image[:,:,3] * image[:,:,2]
    new_image = bgcolor * mask_all + new_image * (1 - mask_all)
    texture_images.append(new_image)
texture_images = np.asarray(texture_images)

how_image = np.zeros((1000, 5000, 3), dtype =np.float32)
for i in range(10):
    show_image[:500, i * 500:i * 500 + 500] = plain_images[i][:,:,:]
    show_image[500:1000, i * 500:i * 500 + 500] = texture_images[i][:,:,:]

import matplotlib.pylab as plt
plt.figure(figsize=(15,20),dpi=200)

plt.imshow(show_image)


np.save("/hdd/Documents/Data/IKEA_PAIR/X_plain_unlabel.npy", plain_images)
np.save("/hdd/Documents/Data/IKEA_PAIR/X_texture_unlabel.npy", texture_images)
