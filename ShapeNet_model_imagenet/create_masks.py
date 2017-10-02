IKEA_plain_directory = "/hdd/Documents/Data/ShapeNetCoreV2/plain_image_random/"
IKEA_texture_directory = "/hdd/Documents/Data/ShapeNetCoreV2/texture_image_random/"

from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
import matplotlib.pylab as plt

IKEA_plain_img_path = sorted([join(IKEA_plain_directory, f) for f in listdir(IKEA_plain_directory) if isfile(join(IKEA_plain_directory, f))])
IKEA_texture_img_path = sorted([join(IKEA_texture_directory, f) for f in listdir(IKEA_texture_directory) if isfile(join(IKEA_texture_directory, f))])

for f, g in zip(IKEA_plain_img_path, IKEA_texture_img_path):

    image_f = misc.imread(f)
    mask_f = np.array(np.product(image_f, axis = 2) == 237 * 237 * 255 * 255).reshape(500,500,1)
    mask_all_f = np.repeat(mask_f, 3, axis = 2)
    image_g = misc.imread(g)
    mask_g = np.array(np.product(image_g, axis = 2) == 237 * 237 * 255 * 255).reshape(500,500,1)
    mask_all_g = np.repeat(mask_g, 3, axis = 2)

    print(f)
    xleft = np.nonzero(np.sum(np.sum(1- mask_all_f, axis = 2), axis = 1))[0][0] - 1
    xright = np.nonzero(np.sum(np.sum(1- mask_all_f, axis = 2), axis = 1))[0][-1] + 1
    ytop = np.nonzero(np.sum(np.sum(1- mask_all_f, axis = 2), axis = 0))[0][0] - 1
    ybottom = np.nonzero(np.sum(np.sum(1- mask_all_f, axis = 2), axis = 0))[0][-1] + 1

    extend = min(int(max(xright - xleft, ybottom - ytop) // 2 * 1.05), 250)
    xleft = 250 - extend
    xright = 250 + extend
    ytop = 250 - extend
    ybottom = 250 + extend
    
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

    new_image_f = new_image_f[xleft:xright, ytop:ybottom]
    mask_all_f = mask_all_f[xleft:xright, ytop:ybottom]

    new_image_g = new_image_g[xleft:xright, ytop:ybottom]
    mask_all_g = mask_all_g[xleft:xright, ytop:ybottom]

    new_image_final_f = 1 - mask_all_f
    new_image_final_g = 1 - mask_all_g

    new_image_final_f = (misc.imresize(new_image_final_f, (32, 32, 3), "nearest") / 255.0 == 1)
    new_image_final_g = (misc.imresize(new_image_final_g, (32, 32, 3), "nearest") / 255.0 == 1)

    misc.imsave("/hdd/Documents/Data/ShapeNetCoreV2/real_plain_image_mask_new/" + f.split("/")[6], new_image_final_f)
    misc.imsave("/hdd/Documents/Data/ShapeNetCoreV2/real_texture_image_mask_new/" + f.split("/")[6], new_image_final_g)
