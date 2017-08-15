IKEA_plain_directory = "/hdd/Documents/Data/ShapeNetCoreV2/plain_image_random/"
IKEA_texture_directory = "/hdd/Documents/Data/ShapeNetCoreV2/texture_image_random/"
back_directory = "/hdd/Documents/Data/ShapeNetCoreV2/bkg_image/road/"

from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
import matplotlib.pylab as plt
import colorsys
from PIL import Image, ImageEnhance

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        value = 128 + factor * (c - 128)
        return max(0, min(255, value))
    return img.point(contrast)

train_data = np.load("/hdd/Documents/Data/ShapeNetCoreV2/cifar10_3class_train.npy")
bkg_img_path = [join(back_directory, f) for f in listdir(back_directory) if isfile(join(back_directory, f))]
base_colors = []
show_image = np.zeros((40 * 3, 40 * 3, 3))
for i in range(5000, 10000):
    base_colors.append(np.array(np.mean(train_data[i, 28:, 28:].reshape(4 * 4, 3), axis = 0), dtype = np.int))
    base_colors.append(np.array(np.mean(train_data[i, 28:, :4].reshape(4 * 4, 3), axis = 0), dtype = np.int))
    base_colors.append(np.array(np.mean(train_data[i, :4, 28:].reshape(4 * 4, 3), axis = 0), dtype = np.int))
    base_colors.append(np.array(np.mean(train_data[i, :4, :4].reshape(4 * 4, 3), axis = 0), dtype = np.int))

for k in range(5629, 10000):
    bkg_image_index = np.random.randint(len(bkg_img_path))
    print(bkg_img_path[bkg_image_index])
    print(Image.open(bkg_img_path[bkg_image_index]))
    contrast_scale = np.random.random() * 0.6 + 0.3
    im = Image.open(bkg_img_path[bkg_image_index])
    contrast = ImageEnhance.Contrast(im)
    bkg_image = contrast.enhance(contrast_scale)
    bkg_image = misc.imresize(bkg_image, (224, 224, 3))
    #plt.imshow(bkg_image)
    #plt.show()
    #bkg_image = misc.imresize(misc.imread(bkg_img_path[bkg_image_index]), (224, 224, 3))
    #bkg_image = misc.imread(bkg_img_path[bkg_image_index])
    
    while(len(bkg_image.shape) == 2 or bkg_image.shape[2] == 4):
        bkg_image_index = np.random.randint(len(bkg_img_path))
        im = Image.open(bkg_img_path[bkg_image_index])
        contrast = ImageEnhance.Contrast(im)
        bkg_image = contrast.enhance(contrast_scale)
        bkg_image = misc.imresize(bkg_image, (224, 224, 3))
        #bkg_image = misc.imresize(change_contrast(Image.open(bkg_img_path[bkg_image_index]), -100), (224, 224, 3))
        #bkg_image = misc.imresize(misc.imread(bkg_img_path[bkg_image_index]), (224, 224, 3))
        #bkg_image = misc.imread(bkg_img_path[bkg_image_index])

    color_scheme_index = np.random.randint(len(base_colors))
    color_scheme = base_colors[color_scheme_index]
    color_scheme_hsv = colorsys.rgb_to_hsv(color_scheme[0] / 255.0,
                                           color_scheme[1] / 255.0,
                                           color_scheme[2] /255.0)

    """
    current_color = None
    if np.random.randint(0, 2) == 0:
        current_color = np.array(np.mean(bkg_image[:4, :4, :3].reshape(4 * 4, 3), axis = 0), dtype = np.int)
    else:
        current_color = np.array(np.mean(bkg_image[-4:, -4:, :3].reshape(4 * 4, 3), axis = 0), dtype = np.int)
    current_color_hsv = colorsys.rgb_to_hsv(current_color[0]/ 255.0, current_color[1] / 255.0, current_color[2] / 255.0)

    difference = np.array(color_scheme_hsv) - np.array(current_color_hsv)
    difference[1] = np.clip(difference[1], -1.0, 0)
    difference[2] = np.clip(difference[2], 0, 0.2)

    print(difference)

    new_image = np.zeros(bkg_image.shape)        
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            new_image[i, j] = np.array(colorsys.rgb_to_hsv(bkg_image[i, j, 0] / 255.0,
                                                           bkg_image[i, j, 1] / 255.0,
                                                           bkg_image[i, j, 2] / 255.0))
            new_image[i, j, 0] = (new_image[i, j, 0] + difference[0]) % 1
            new_image[i, j, 1] = np.clip(new_image[i, j, 1] + difference[1], 0, 1)
            new_image[i, j, 2] = np.clip(new_image[i, j, 2] + difference[1], 0, 1)
            new_image[i, j] = colorsys.hsv_to_rgb(new_image[i, j, 0],
                                                  new_image[i, j, 1],
                                                  new_image[i, j, 2])
    """
    new_image_2 = np.repeat(np.mean(np.array(bkg_image), axis = 2).reshape(224, 224, 1), 3, axis = 2)
    current_color = None
    if np.random.randint(0, 2) == 0:
        current_color = np.array(np.mean(new_image_2[:4, :4, :3].reshape(4 * 4, 3), axis = 0), dtype = np.int)
    else:
        current_color = np.array(np.mean(new_image_2[-4:, -4:, :3].reshape(4 * 4, 3), axis = 0), dtype = np.int)
    current_color_hsv = colorsys.rgb_to_hsv(current_color[0]/ 255.0, current_color[1] / 255.0, current_color[2] / 255.0)

    difference = np.array(color_scheme_hsv) - np.array(current_color_hsv)
    
    for i in range(new_image_2.shape[0]):
        for j in range(new_image_2.shape[1]):
            new_image_2[i, j] = np.array(colorsys.rgb_to_hsv(new_image_2[i, j, 0] / 255.0,
                                                           new_image_2[i, j, 1] / 255.0,
                                                           new_image_2[i, j, 2] / 255.0))
            new_image_2[i, j, 0] = (new_image_2[i, j, 0] + difference[0]) % 1
            new_image_2[i, j, 1] = np.clip(new_image_2[i, j, 1] + difference[1], 0, 1)
            new_image_2[i, j, 2] = np.clip(new_image_2[i, j, 2] + difference[1], 0, 1)
            new_image_2[i, j] = colorsys.hsv_to_rgb(new_image_2[i, j, 0],
                                                  new_image_2[i, j, 1],
                                                  new_image_2[i, j, 2])
    

    #new_image *= 255.0
    #new_image = np.array(new_image, dtype = np.uint8)
    new_image_2 *= 255.0
    new_image_2 = np.array(new_image_2, dtype = np.uint8)
    bkg_image = np.array(bkg_image, dtype = np.uint8)

    #misc.imsave("/hdd/Documents/Data/shapeNetTexture_bkg/change_bkg_image_%d.png" %k, new_image)
    misc.imsave("/hdd/Documents/Data/shapeNetTexture_bkg_road/original_bkg_image_%d.png" %k, bkg_image)
    misc.imsave("/hdd/Documents/Data/shapeNetTexture_bkg_road/grey_bkg_image_%d.png" %k, new_image_2)





