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

    """
    if f.split('_')[2] != "random/02958343":
        continue
    """
    image_f = misc.imread(f)
    if len(image_f[np.product(image_f, axis = 2) == 237 * 237 * 255 * 255]) == 0:
        continue
    
    mask_f = np.array(np.product(image_f, axis = 2) == 237 * 237 * 255 * 255).reshape(500,500,1)
    mask_all_f = np.repeat(mask_f, 3, axis = 2)
    
    image_f[np.product(image_f, axis = 2) == 237 * 237 * 255 * 255] = np.array([237, 237, 255, 0])
    image_g = misc.imread(g)
    image_g[np.product(image_g, axis = 2) == 237 * 237 * 255 * 255] = np.array([237, 237, 255, 0])

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

    image_f = image_f[xleft:xright, ytop:ybottom]
    image_g = image_g[xleft:xright, ytop:ybottom]

    image_f = misc.imresize(image_f, (32, 32, 4), "bilinear")
    image_g = misc.imresize(image_g, (32, 32, 4), "bilinear")

    image_f[image_f[:,:,3] < 100, 3] = 0
    image_f[image_f[:,:,3] >= 100, 3] = 255
    image_g[image_g[:,:,3] < 100, 3] = 0
    image_g[image_g[:,:,3] >= 100, 3] = 255

    bgcolor = np.array([237, 237, 255]).reshape(1, 1, 3)

    new_image_f = np.zeros((32, 32, 3))
    new_image_f[:,:,0] = (1 - image_f[:,:,3]) * image_f[:,:,0] + image_f[:,:,3] * image_f[:,:,0]
    new_image_f[:,:,1] = (1 - image_f[:,:,3]) * image_f[:,:,1] + image_f[:,:,3] * image_f[:,:,1]
    new_image_f[:,:,2] = (1 - image_f[:,:,3]) * image_f[:,:,2] + image_f[:,:,3] * image_f[:,:,2]
    new_image_g = np.zeros((32, 32, 3))
    new_image_g[:,:,0] = (1 - image_g[:,:,3]) * image_g[:,:,0] + image_g[:,:,3] * image_g[:,:,0]
    new_image_g[:,:,1] = (1 - image_g[:,:,3]) * image_g[:,:,1] + image_g[:,:,3] * image_g[:,:,1]
    new_image_g[:,:,2] = (1 - image_g[:,:,3]) * image_g[:,:,2] + image_g[:,:,3] * image_g[:,:,2]

    mask_all_f = (image_f[:,:,3]!=0).reshape(32, 32, 1)
    mask_all_g = (image_g[:,:,3]!=0).reshape(32, 32, 1)

    bkg_image = bgcolor

    new_image_final_f = bkg_image * (1 - mask_all_f) + new_image_f * mask_all_f
    new_image_final_g = bkg_image * (1 - mask_all_g) + new_image_g * mask_all_g

    misc.imsave("/hdd/Documents/Data/ShapeNetCoreV2/non_real_plain_image_small_v2/" + f.split("/")[6], new_image_final_f)
    misc.imsave("/hdd/Documents/Data/ShapeNetCoreV2/non_real_texture_image_small_v2/" + f.split("/")[6], new_image_final_g)
    misc.imsave("/hdd/Documents/Data/ShapeNetCoreV2/non_real_plain_image_small_v2_mask/" + f.split("/")[6], mask_all_f.reshape(32, 32))
    misc.imsave("/hdd/Documents/Data/ShapeNetCoreV2/non_real_texture_image_small_v2_mask/" + f.split("/")[6], mask_all_g.reshape(32, 32))






