# !/usr/bin/env python
import os
import pdb


def link_dataset():
   json_folder = '/data/user/data/breezecheng/dataset/image_detection/object_detect_catall_20190625/annotations_cat39'
   img_folder = '/data/user/data/breezecheng/dataset/image_detection/object_detect_catall_20190625'
   save_folder = '../datasets/full_product_det'
   os.system('ln -s {}/annotations_cat39 {}/annotations'.format(img_folder, save_folder))
   for json_file in os.listdir(json_folder):
       if json_file.endswith('.json'):
          json_name = json_file[:-5]
          print(json_name)
          os.system('ln -s {}/JPEGImages/{} {}/{}'.format(img_folder, json_name, save_folder, json_name))

def config_dataset():
    json_files = os.listdir("/data/user/data/breezecheng/dataset/image_detection/object_detect_catall_20190625/annotations_cat39/")
    print('{} json files'.format(len(json_files)))
    fid = open('xx', 'w')
    for json_file in json_files:
        if json_file.endswith('.json'):
           fid.write('"coco_%s",' % (json_file[:-5]))
    fid.close()

def path_dataset():
    json_files = os.listdir("/data/user/data/breezecheng/dataset/image_detection/object_detect_catall_20190625/annotations_cat39")
    fid = open('xx', 'w')
    for json_file in json_files:
        if json_file.endswith('.json'):
           fid.write('"coco_%s": {\n' % (json_file[:-5]))
           fid.write('  "img_dir": "full_product_det/%s",\n' % (json_file[:-5]))
           fid.write('  "ann_file": "full_product_det/annotations/%s"\n' % (json_file))
           fid.write('},\n')
    fid.close()

if __name__ == '__main__':
    path_dataset()
