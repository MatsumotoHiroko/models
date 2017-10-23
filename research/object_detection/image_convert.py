import os, sys
import re

import config
from PIL import Image

import argparse

# arg
parser = argparse.ArgumentParser()
parser.add_argument(
  '--target_dir',
  type=str,
  default='dog,cat,sheep',
  help='image files target directry.'
)
FLAGS, unparsed = parser.parse_known_args()

output_ext = 'png'
target_dirs = FLAGS.target_dir.split(',')
BLACK = R, G, B, A = (0, 0, 0, 255)

for dirpath, dirnames, filenames in os.walk(config.CROP_IMAGES_DIR):
    for dirname in dirnames:
        if dirname not in target_dirs:
            continue
        member_dir = os.path.join(dirpath, dirname)
        convert_dir = os.path.join(config.CONVERT_IMAGES_DIR, dirname)
        print("##### member_dir")
        print(member_dir)
        print("##### convert_dir")
        print(convert_dir)
        if not os.path.isdir(convert_dir):
            os.makedirs(convert_dir)
        for dirpath2, dirnames2, filenames2 in os.walk(member_dir):
            index = 0
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
                    if im.mode != "RGB":
                        im = im.convert("RGB") # RGBモードに変換する
                    im.thumbnail(config.CONVERT_IMAGE_SIZE, Image.ANTIALIAS)        

                    print("###### convert file")
                    print(os.path.join(convert_dir, outfile))
                    if im is not None:
                        bg = Image.new("RGBA", config.CONVERT_IMAGE_SIZE, BLACK)
                        img_width, img_height = im.size
                        center = (config.CONVERT_IMAGE_SIZE[0] - img_width) // 2, (config.CONVERT_IMAGE_SIZE[1] - img_height) // 2
                        bg.paste(im, center)
                        bg.save(os.path.join(convert_dir, outfile), 'PNG')        
                        index += 1
                    else:
                        print("save error")
            print('convert:{} / {}files'.format(dirname, index)) 
