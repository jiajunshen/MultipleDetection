import sys
import os
from create_image_pairs import create_image_pairs

database = "IKEA"
database_path = os.environ[database]
obj_file = open(database_path + "/all_obj.txt", "r")
content = obj_file.readlines()
content = [x.strip() for x in content]


content = [content[i][8:-4] for i in range(len(content))]
furniture_type = []
furniture_obj = []
for item in content:
    item_type, item_obj = item.split("/")
    furniture_type.append(item_type)
    furniture_obj.append(item_obj)

obj_path_list = []
save_path_list = []
for item_type, item_obj in zip(furniture_type, furniture_obj):
    folder_name = "nojitter_png_" + item_obj[:-11]
    obj_file_path = os.path.join(database_path, item_type, folder_name, item_obj)
    obj_path_list.append(obj_file_path)
    save_path_list.append(item_type + "_" + item_obj)

create_image_pairs(obj_path_list, save_path_list)
