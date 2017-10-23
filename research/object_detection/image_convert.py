import os, sys
import re

import config
from PIL import Image

output_ext = 'png'
for dirpath, dirnames, filenames in os.walk(config.CROP_IMAGES_DIR):
    for dirname in dirnames:
        member_dir = os.path.join(dirpath, dirname)
        convert_dir = os.path.join(config.CONVERT_IMAGES_DIR, dirname)
        print("##### member_dir")
        print(member_dir)
        print("##### convert_dir")
        print(convert_dir)
        if not os.path.isdir(convert_dir):
            os.makedirs(convert_dir)
        for dirpath2, dirnames2, filenames2 in os.walk(member_dir):
            for infile in filenames2:
                (fn,ext) = os.path.splitext(infile)
                outfile = fn + '.{}'.format(output_ext)
                print("##### infile")
                print(infile)
                print("##### outfile")
                print(outfile)
                if ext.upper() in config.exts:
                    im = Image.open(os.path.join(dirpath, dirname, infile))
                    # image type convert        
                    if outfile != infile:
                        if im.mode != "RGB":
                            im = im.convert("RGB") # RGBモードに変換する
                    im = im.resize(config.CONVERT_IMAGE_SIZE)        
                    print("###### convert file")
                    print(os.path.join(convert_dir, outfile))
                    im.save(os.path.join(convert_dir, outfile))        
