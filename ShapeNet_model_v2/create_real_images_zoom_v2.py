IKEA_texture_directory = "/hdd/Documents/Data/shapeNetTexture_boat/"
back_directory = "/hdd/Documents/Data/shapeNetTexture_bkg/"

from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
import matplotlib.pylab as plt

IKEA_texture_img_path = sorted([join(IKEA_texture_directory, f) for f in listdir(IKEA_texture_directory) if isfile(join(IKEA_texture_directory, f))])
bkg_img_path = [join(back_directory, f) for f in listdir(back_directory) if isfile(join(back_directory, f))]
print(IKEA_texture_img_path)
count = 0
for f in IKEA_texture_img_path:
    count += 1
    """
    if f.split('_')[2] != "random/02958343":
        continue
    """
    image_f = misc.imread(f)
     
    if len(image_f[np.product(image_f, axis = 2) == 229 * 229 * 255]) == 0:
        print("alert")
        print(np.product(image_f, axis = 2)[0,0])
        continue
    
    mask_f = np.array(np.product(image_f, axis = 2) == 229 * 229 * 255).reshape(224,224,1)
    mask_all_f = np.repeat(mask_f, 3, axis = 2)
    
    bkg_image_index = np.random.randint(len(bkg_img_path))
    bkg_image = misc.imread(bkg_img_path[bkg_image_index])

    combine_image = bkg_image * mask_all_f + image_f * (1 - mask_all_f)

    image_f = misc.imresize(combine_image, (32, 32, 3), "bilinear")
    
    bgcolor = np.array([255, 255, 255]).reshape(1, 1, 3)

    new_image_final_f = np.array(image_f, dtype=np.uint8)

    misc.imsave("/hdd/Documents/Data/shapeNetTexture_small_real_new_boat/" + f.split("/")[5], new_image_final_f)
    #misc.imsave("/hdd/Documents/Data/shapeNetTexture_mask_small/" + f.split("/")[5], mask_all_f.reshape(32, 32))
