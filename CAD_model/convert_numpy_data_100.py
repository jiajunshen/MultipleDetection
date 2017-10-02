import numpy as np
from os import listdir
from os.path import isfile, join
plain_file_path = "/hdd/Documents/Data/ShapeNetCoreV2/plain_image_random"
plain_files = [join(plain_file_path, f) for f in listdir(plain_file_path) if isfile(join(plain_file_path, f))]
texture_file_path = '/hdd/Documents/Data/ShapeNetCoreV2/texture_image_random'
texture_files = [join(texture_file_path, f) for f in listdir(texture_file_path) if isfile(join(texture_file_path, f))]

# Source => Target = (BGColor + Source) =
# Target.R = ((1 - Source.A) * BGColor.R) + (Source.A * Source.R)
# Target.G = ((1 - Source.A) * BGColor.G) + (Source.A * Source.G)
# Target.B = ((1 - Source.A) * BGColor.B) + (Source.A * Source.B)

from scipy import misc
plain_images = []
plain_images_tmp = []
for f in plain_files:
    image = misc.imread(f)
    mask = np.array(np.product(image, axis = 2) == 237 * 237 * 255 * 255).reshape(100,100,1)
    mask_all = np.repeat(mask, 3, axis = 2)
    
    image = image / 255.0
    bgcolor = np.array([237, 237, 255]).reshape(1, 1, 3) / 255.0
    new_image = np.zeros((100, 100, 3))
    new_image[:,:,0] = (1 - image[:,:,3]) * image[:,:,0] + image[:,:,3] * image[:,:,0]
    new_image[:,:,1] = (1 - image[:,:,3]) * image[:,:,1] + image[:,:,3] * image[:,:,1]
    new_image[:,:,2] = (1 - image[:,:,3]) * image[:,:,2] + image[:,:,3] * image[:,:,2]
    new_image = bgcolor * mask_all + new_image * (1 - mask_all)
    if len(plain_images_tmp) < 100:
        plain_images_tmp.append(new_image)
    else:
        plain_images.append(np.asarray(plain_images_tmp, dtype=np.float32))
        plain_images_tmp = []

if len(plain_images_tmp) != 0:
    plain_images.append(np.asarray(plain_images_tmp, dtype=np.float32))

plain_images = np.vstack(plain_images)
np.save("/hdd/Documents/Data/IKEA_PAIR/X_plain_unlabel_random.npy", plain_images)
plain_images_sample = plain_images[:10]
del plain_images

texture_images = []
texture_images_tmp = []
for f in texture_files:
    image = misc.imread(f)
    mask = np.array(np.product(image, axis = 2) == 237 * 237 * 255 * 255).reshape(100,100,1)
    mask_all = np.repeat(mask, 3, axis = 2)
    
    image = image / 255.0
    bgcolor = np.array([237, 237, 255]).reshape(1, 1, 3) / 255.0
    new_image = np.zeros((100, 100, 3))
    new_image[:,:,0] = (1 - image[:,:,3]) * image[:,:,0] + image[:,:,3] * image[:,:,0]
    new_image[:,:,1] = (1 - image[:,:,3]) * image[:,:,1] + image[:,:,3] * image[:,:,1]
    new_image[:,:,2] = (1 - image[:,:,3]) * image[:,:,2] + image[:,:,3] * image[:,:,2]
    new_image = bgcolor * mask_all + new_image * (1 - mask_all)
    if len(texture_images_tmp) < 100:
        texture_images_tmp.append(new_image)
    else:
        texture_images.append(np.asarray(texture_images_tmp, dtype=np.float32))
        texture_images_tmp = []

if len(texture_images_tmp) != 0:
    texture_images.append(np.asarray(texture_images_tmp, dtype=np.float32))

texture_images = np.vstack(texture_images)

np.save("/hdd/Documents/Data/IKEA_PAIR/X_texture_unlabel_random.npy", texture_images)
texture_images_sample = texture_images[:10]
del texture_images

show_image = np.zeros((200, 1000, 3), dtype =np.float32)
for i in range(10):
    show_image[:100, i * 100 : i * 100 + 100] = plain_images_sample[i][:,:,:]
    show_image[100:200, i * 100 : i * 100 + 100] = texture_images_sample[i][:,:,:]

import matplotlib.pylab as plt
plt.figure(figsize=(15,20),dpi=200)

plt.imshow(show_image)


