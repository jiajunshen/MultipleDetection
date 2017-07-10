import bpy
import sys
from os import listdir
from os.path import isfile, join
from os.path import isdir

root_path = "/hdd/Documents/Data/ShapeNetCoreV2/02691156"

file_path_list = [join(root_path, f) for f in listdir(root_path) \
                  if isdir(join(root_path, f))]

for file_path in file_path_list:

    bpy.ops.import_scene.obj(filepath=file_path + "/models/model_normalized.obj")

    bpy.ops.export_scene.obj(filepath=file_path + "/models/model_new.obj")

    bpy.ops.object.delete()
