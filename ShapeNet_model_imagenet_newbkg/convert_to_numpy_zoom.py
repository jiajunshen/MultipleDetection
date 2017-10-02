from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
import matplotlib.pylab as plt


items = ["plane", "car", "boat", "sofa", "table", "bus", "chair", "display", "gun"]

for item in items:
    texture_dataset_path = "/hdd/Documents/Data/shapeNetTexture_small_real_10class_newbkg/shapeNetTexture_small_realbackground_%s/" %(item)
    #texture_dataset_path = "/hdd/Documents/Data/shapeNetTexture_small_mask_10class/shapeNetTexture_mask_small_%s/" %(item)
    #texture_dataset_path = "/hdd/Documents/Data/shapeNetTexture_mask_small_boat/"
    #plain_dataset_path = "/hdd/Documents/Data/ShapeNetCoreV2/real_plain_image_v2/"
    #texture_dataset_path = "/hdd/Documents/Data/ShapeNetCoreV2/real_texture_image_v2/"
    #plain_dataset_path = "/hdd/Documents/Data/ShapeNetCoreV2/non_real_plain_image_small/"
    #texture_dataset_path = "/hdd/Documents/Data/ShapeNetCoreV2/non_real_texture_image_small/"

    texture_path_list = sorted([join(texture_dataset_path, f) for f in listdir(texture_dataset_path) if isfile(join(texture_dataset_path, f))])

    #label = []
    texture_all_data = []
    #all_label = [f.split("_")[5] for f in plain_path_list]
    #print(all_label)
    #all_classes = np.unique(np.array(all_label)).tolist()
    for f in texture_path_list:
        image_f = misc.imread(f)
        #label.append(all_classes.index(l))
        texture_all_data.append(image_f)
    #np.save("/hdd/Documents/Data/ShapeNetCoreV1/white_background_%s.npy" %(item), texture_all_data)
    np.save("/hdd/Documents/Data/ShapeNetCoreV1/real_background_newbkg_%s.npy" %(item), texture_all_data)
    #np.save("/hdd/Documents/Data/ShapeNetCoreV1/small_mask_%s.npy" %(item), texture_all_data)
    #np.save("/hdd/Documents/Data/ShapeNetCoreV1/real_background_whitebackground_%s.npy" %(item), texture_all_data)
    #np.save("/hdd/Documents/Data/ShapeNetCoreV1/real_background_new_boat_mask.npy", texture_all_data)
    #np.save("/hdd/Documents/Data/ShapeNetCoreV2/ShapeNet_NonSynthetic_plain_v2.npy", plain_all_data)
    #np.save("/hdd/Documents/Data/ShapeNetCoreV2/ShapeNet_NonSynthetic_texture_v2.npy", texture_all_data)
    #np.save("/hdd/Documents/Data/ShapeNetCoreV2/ShapeNet_Synthetic_plain_mask.npy", plain_all_data)
    #np.save("/hdd/Documents/Data/ShapeNetCoreV2/ShapeNet_Synthetic_texture_mask.npy", texture_all_data)
    #np.save("/hdd/Documents/Data/ShapeNetCoreV2/ShapeNet_Synthetic_label.npy", label)
    #print(label)

