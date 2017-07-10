IKEA_directory = "./CAD_Plain/"
back_directory = "/Users/jiajunshen/Documents/Research/recurrent-spatial-transformer-code/cnnDetection/bkgimage/empty_room_bkg/"

from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc

IKEA_img_path = [join(IKEA_directory, f) for f in listdir(IKEA_directory) if isfile(join(IKEA_directory, f))]
bkg_img_path = [join(back_directory, f) for f in listdir(back_directory) if isfile(join(back_directory, f))]

final_images = []
for f in IKEA_img_path:
    image = misc.imread(f)
    mask = np.array(np.product(image, axis = 2) == 237 * 237 * 255 * 255).reshape(500,500,1)
    mask_all = np.repeat(mask, 3, axis = 2)
    
    image = image / 255.0
    bgcolor = np.array([237, 237, 255]).reshape(1, 1, 3) / 255.0
    new_image = np.zeros((500, 500, 3))
    new_image[:,:,0] = (1 - image[:,:,3]) * image[:,:,0] + image[:,:,3] * image[:,:,0]
    new_image[:,:,1] = (1 - image[:,:,3]) * image[:,:,1] + image[:,:,3] * image[:,:,1]
    new_image[:,:,2] = (1 - image[:,:,3]) * image[:,:,2] + image[:,:,3] * image[:,:,2]
    
    bkg_image_index = np.random.randint(len(bkg_img_path))
    bkg_image = misc.imresize(misc.imread(bkg_img_path[bkg_image_index]), (500,500,3)) / 255.0
    new_image_final = bkg_image * mask_all + new_image * (1 - mask_all)
    final_images.append(new_image_final)
final_images = np.asarray(final_images)
