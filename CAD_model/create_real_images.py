IKEA_plain_directory = "/hdd/Documents/Data/IKEA_PAIR/CAD_Plain/"
IKEA_texture_directory = "/hdd/Documents/Data/IKEA_PAIR/CAD_Texture/"
back_directory = "/hdd/Documents/Data/empty_room_bkg/"

from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
import matplotlib.pylab as plt

IKEA_plain_img_path = sorted([join(IKEA_plain_directory, f) for f in listdir(IKEA_plain_directory) if isfile(join(IKEA_plain_directory, f))])
IKEA_texture_img_path = sorted([join(IKEA_texture_directory, f) for f in listdir(IKEA_texture_directory) if isfile(join(IKEA_texture_directory, f))])

file_label = [f.split('_')[1] for f in listdir(IKEA_plain_img_path)]
all_classes = np.unique(file_label).tolist()

bkg_img_path = [join(back_directory, f) for f in listdir(back_directory) if isfile(join(back_directory, f))]

final_images_plain = []
final_images_texture = []

for f, g in zip(IKEA_plain_img_path, IKEA_texture_img_path):
    image_f = misc.imread(f)
    mask_f = np.array(np.product(image_f, axis = 2) == 237 * 237 * 255 * 255).reshape(500,500,1)
    mask_all_f = np.repeat(mask_f, 3, axis = 2)
    image_g = misc.imread(g)
    mask_g = np.array(np.product(image_g, axis = 2) == 237 * 237 * 255 * 255).reshape(500,500,1)
    mask_all_g = np.repeat(mask_g, 3, axis = 2)
    
    image_f = image_f / 255.0
    image_g = image_g / 255.0

    bgcolor = np.array([237, 237, 255]).reshape(1, 1, 3) / 255.0

    new_image_f = np.zeros((500, 500, 3))
    new_image_f[:,:,0] = (1 - image_f[:,:,3]) * image_f[:,:,0] + image_f[:,:,3] * image_f[:,:,0]
    new_image_f[:,:,1] = (1 - image_f[:,:,3]) * image_f[:,:,1] + image_f[:,:,3] * image_f[:,:,1]
    new_image_f[:,:,2] = (1 - image_f[:,:,3]) * image_f[:,:,2] + image_f[:,:,3] * image_f[:,:,2]
    new_image_g = np.zeros((500, 500, 3))
    new_image_g[:,:,0] = (1 - image_g[:,:,3]) * image_g[:,:,0] + image_g[:,:,3] * image_g[:,:,0]
    new_image_g[:,:,1] = (1 - image_g[:,:,3]) * image_g[:,:,1] + image_g[:,:,3] * image_g[:,:,1]
    new_image_g[:,:,2] = (1 - image_g[:,:,3]) * image_g[:,:,2] + image_g[:,:,3] * image_g[:,:,2]
    
    bkg_image_index = np.random.randint(len(bkg_img_path))
    bkg_image = misc.imresize(misc.imread(bkg_img_path[bkg_image_index]), (500,500,3)) / 255.0

    new_image_final_f = bkg_image * mask_all_f + new_image_f * (1 - mask_all_f)
    new_image_final_g = bkg_image * mask_all_g + new_image_g * (1 - mask_all_g)

    final_images_plain.append(new_image_final_f)
    final_images_texture.append(new_image_final_g)

final_images_plain = np.asarray(final_images_plain, dtype=np.float32)
final_images_texture = np.asarray(final_images_texture, dtype=np.float32)

training_index = []
testing_index = []
for item in all_classes:
    index_list = np.where(np.asarray(file_label) == item)[0].tolist()
    amount = len(index_list)
    training_index += index_list[:amount // 3 * 2]
    testing_index += index_list[amount // 3 * 2:]

training_plain_img = final_image_plain[training_index]
testing_plain_img = final_image_plain[testing_index]

training_texture_img = final_images_texture[training_index]
testing_texture_img = final_images_texture[testing_index]

np.save("/hdd/Documents/Data/IKEA_PAIR/X_plain_real_train.npy", training_plain_img)
np.save("/hdd/Documents/Data/IKEA_PAIR/X_texture_real_train.npy", training_texture_img)
np.save("/hdd/Documents/Data/IKEA_PAIR/X_plain_real_test.npy", testing_plain_img)
np.save("/hdd/Documents/Data/IKEA_PAIR/X_texture_real_test.npy", testing_texture_img)
np.save("/hdd/Documents/Data/IKEA_PAIR/Y_train_real.npy", np.asarray(file_label)[training_index])
np.save("/hdd/Documents/Data/IKEA_PAIR/Y_test_real.npy", np.asarray(file_label)[testing_index])



