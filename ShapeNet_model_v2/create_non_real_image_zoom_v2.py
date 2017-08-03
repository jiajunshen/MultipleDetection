IKEA_texture_directory = "/hdd/Documents/Data/shapeNetTexture_boat/"

from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
import matplotlib.pylab as plt

IKEA_texture_img_path = sorted([join(IKEA_texture_directory, f) for f in listdir(IKEA_texture_directory) if isfile(join(IKEA_texture_directory, f))])
print(IKEA_texture_img_path)
count = 0
for f in IKEA_texture_img_path:
    count += 1
    image_f_original = np.ones((224, 224, 4)) * 255
    image_f = misc.imread(f)
    image_f_original[:, :, :3] = image_f
    image_f = image_f_original
     
    if len(image_f[np.product(image_f, axis = 2) == 229 * 229 * 255 * 255]) == 0:
        print("alert")
        print(np.product(image_f, axis = 2)[0,0])
        continue
    
    mask_f = np.array(np.product(image_f, axis = 2) == 229 * 229 * 255 * 255).reshape(224,224,1)
    mask_all_f = np.repeat(mask_f, 3, axis = 2)
    
    image_f[np.product(image_f, axis = 2) == 229 * 229 * 255 * 255] = np.array([229, 229, 255, 0])
    
    
    image_f = misc.imresize(image_f, (32, 32, 4), "bilinear") / 255.0
    
    bgcolor = np.array([255, 255, 255]).reshape(1, 1, 3) / 255.0

    new_image_f = np.zeros((32, 32, 3))
    new_image_f[:,:,0] = (1 - image_f[:,:,3]) * image_f[:,:,0] + image_f[:,:,3] * image_f[:,:,0]
    new_image_f[:,:,1] = (1 - image_f[:,:,3]) * image_f[:,:,1] + image_f[:,:,3] * image_f[:,:,1]
    new_image_f[:,:,2] = (1 - image_f[:,:,3]) * image_f[:,:,2] + image_f[:,:,3] * image_f[:,:,2]

    mask_all_f = (image_f[:,:,3]!=0).reshape(32, 32, 1)
    """ 
    bkg_image_index = np.random.randint(len(bkg_img_path))
    print(bkg_img_path[bkg_image_index])
    bkg_image = misc.imresize(misc.imread(bkg_img_path[bkg_image_index]), (32, 32, 3))
    while(bkg_image.shape[2] == 4):
        bkg_image_index = np.random.randint(len(bkg_img_path))
        bkg_image = misc.imresize(misc.imread(bkg_img_path[bkg_image_index]), (32, 32, 3))
    """
    bkg_image = bgcolor
    
    new_image_final_f = bkg_image * (1 - mask_all_f) + new_image_f * mask_all_f

    new_image_final_f *= 255.0

    new_image_final_f = np.array(new_image_final_f, dtype=np.uint8)

    #misc.imsave("/hdd/Documents/Data/shapeNetTexture_small/" + f.split("/")[5], new_image_final_f)
    misc.imsave("/hdd/Documents/Data/shapeNetTexture_mask_small_boat/" + f.split("/")[5], mask_all_f.reshape(32, 32))






