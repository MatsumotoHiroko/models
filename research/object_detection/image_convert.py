import os, sys
import re

import config
from PIL import Image

output_ext = 'png'
for dirpath, dirnames, filenames in os.walk(CROP_IMAGES_DIR):
    for dirname in dirnames:
        member_dir = os.path.join(dirpath, dirname)
        convert_dir = os.path.join(PATH_TO_CROP_IMAGES_DIR, dirname)
        print("##### member_dir")
        print(member_dir)
        print("##### convert_dir")
        print(convert_dir)
        if not os.path.isdir(convert_dir):
            os.makedirs(convert_dir)
        for dirpath2, dirnames2, filenames2 in os.walk(member_dir):
            if infile in filenames2:
                (fn,ext) = os.path.splitext(infile)
                outfile = fn + '.{}'.format(output_ext)
                print("##### infile")
                print(infile)
                print("##### outfile")
                print(outfile)
                if ext.upper() in exts:
                    im = Image.open(os.path.join(CROP_IMAGES_DIR, dirpath, infile)
                    # image type convert        
                    if outfile != infile:
                        if im.mode != "RGB":
                            im = im.convert("RGB")　# RGBモードに変換する
                    im.resize(CONVERT_IMAGE_SIZE)        
                    #im.save(os.path.join(convert_dir, outfile))        
