import os
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
import matplotlib.pylab as plt

back_directory = "/hdd/Documents/Data/bkg_flickr/"
bkg_img_path = [join(back_directory, f) for f in listdir(back_directory) if isfile(join(back_directory, f))]

for i in range(len(bkg_img_path)):
        bkg_image_index = i
        bkg_image = misc.imread(bkg_img_path[bkg_image_index])
        """
        print(bkg_img_path[i])
        print(bkg_image.shape)

        while(len(bkg_image.shape) == 2 or bkg_image.shape[2] == 4):
            bkg_image_index = np.random.randint(len(bkg_img_path))
            bkg_image = misc.imread(bkg_img_path[bkg_image_index])
        """
        if len(bkg_image.shape) != 3:
            print bkg_image.shape
            print bkg_img_path[bkg_image_index]
