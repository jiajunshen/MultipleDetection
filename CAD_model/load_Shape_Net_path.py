import sys
import os
from create_image_pairs import create_image_pairs
from os import listdir
from os.path import isfile, join, isdir
database = "SHAPENET"

folder_path_list = []
folder_name_list = []
all_object_class = []
object_list = ["02808440", "03207941", "03642806", "03691469", "03761084", "04515003"]
for object_name in object_list:
    database_path = join(os.environ[database], object_name)
    folder_path_list += [join(database_path, directory) for directory in listdir(database_path) if isdir(join(database_path, directory))]
    folder_name_list += [directory for directory in listdir(database_path) if isdir(join(database_path, directory))]
    all_object_class += (len(folder_name_list) - len(all_object_class)) * [object_name]
obj_path_list = []
save_path_list = []

i = 0
for folder_path, folder_name, object_name in zip(folder_path_list, folder_name_list, all_object_class):
    obj_path_list.append(join(folder_path, "models/model_new.obj"))
    save_path_list.append(object_name + "_" + folder_name)
    i+=1
    if i > 3:
        break

create_image_pairs(obj_path_list, save_path_list)
