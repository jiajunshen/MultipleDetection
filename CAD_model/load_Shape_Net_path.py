import sys
import os
from create_image_pairs import create_image_pairs
from os import listdir
from os.path import isfile, join, isdir
database = "SHAPENET"
database_path = join(os.environ[database], "02691156")
folder_path_list = [join(database_path, directory) for directory in listdir(database_path) if isdir(join(database_path, directory))]
folder_name_list = [directory for directory in listdir(database_path) if isdir(join(database_path, directory))]

obj_path_list = []
save_path_list = []

i = 0
for folder_path, folder_name in zip(folder_path_list, folder_name_list):
    obj_path_list.append(join(folder_path, "models/model_new.obj"))
    save_path_list.append("02691156_" + folder_name)
    i+=1
    if i > 3:
        break

create_image_pairs(obj_path_list, save_path_list)
