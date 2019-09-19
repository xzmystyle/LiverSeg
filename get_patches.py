import py_wsi
import os

file_dir = "./data/XML/"
db_location = "./data/Label_patches"
xml_dir = file_dir
patch_size = 256
level = 16
db_name = "patch_db"
overlap = 0
# All possible labels mapped to integer ids in order of increasing severity.
label_map = {'Normal':0,'Viable Tumor':1,'Negative Pen':2}

if not os.path.exists(db_location):
    os.mkdir(db_location)

turtle = py_wsi.Turtle(file_dir, db_location, db_name, storage_type = 'disk',xml_dir = xml_dir, label_map=label_map)

print("Patch size:", patch_size)
turtle.sample_and_store_patches(patch_size, level, overlap , load_xml=True, limit_bounds=True)

