from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
import matplotlib.pylab as plt


plain_dataset_path = "/hdd/Documents/Data/ShapeNetCoreV2/real_plain_image_small_v2/"
texture_dataset_path = "/hdd/Documents/Data/ShapeNetCoreV2/real_texture_image_small_v2/"
#plain_dataset_path = "/hdd/Documents/Data/ShapeNetCoreV2/real_plain_image_v2/"
#texture_dataset_path = "/hdd/Documents/Data/ShapeNetCoreV2/real_texture_image_v2/"
#plain_dataset_path = "/hdd/Documents/Data/ShapeNetCoreV2/non_real_plain_image_small/"
#texture_dataset_path = "/hdd/Documents/Data/ShapeNetCoreV2/non_real_texture_image_small/"

plain_path_list = sorted([join(plain_dataset_path, f) for f in listdir(plain_dataset_path) if isfile(join(plain_dataset_path, f))])
texture_path_list = sorted([join(texture_dataset_path, f) for f in listdir(texture_dataset_path) if isfile(join(texture_dataset_path, f))])

plain_all_data = []
texture_all_data = []
label = []

all_label = [f.split("_")[5] for f in plain_path_list]
print(all_label)
all_classes = np.unique(np.array(all_label)).tolist()
for f, g, l in zip(plain_path_list, texture_path_list, all_label):
    image_f = misc.imread(f)
    image_g = misc.imread(g)
    label.append(all_classes.index(l))
    plain_all_data.append(image_f)
    texture_all_data.append(image_g)
np.save("/hdd/Documents/Data/ShapeNetCoreV2/ShapeNet_Synthetic_plain_v2.npy", plain_all_data)
np.save("/hdd/Documents/Data/ShapeNetCoreV2/ShapeNet_Synthetic_texture_v2.npy", texture_all_data)
#np.save("/hdd/Documents/Data/ShapeNetCoreV2/ShapeNet_NonSynthetic_plain_v2.npy", plain_all_data)
#np.save("/hdd/Documents/Data/ShapeNetCoreV2/ShapeNet_NonSynthetic_texture_v2.npy", texture_all_data)
#np.save("/hdd/Documents/Data/ShapeNetCoreV2/ShapeNet_Synthetic_plain_mask.npy", plain_all_data)
#np.save("/hdd/Documents/Data/ShapeNetCoreV2/ShapeNet_Synthetic_texture_mask.npy", texture_all_data)
#np.save("/hdd/Documents/Data/ShapeNetCoreV2/ShapeNet_Synthetic_label.npy", label)
print(label)

