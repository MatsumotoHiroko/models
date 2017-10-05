#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import re

import config

if __name__ == '__main__':
  outdir = sys.argv[1]

  if not os.path.isdir(outdir):
    sys.exit('%s is not directory' % outdir)


  exts = ['.JPG','.JPEG', '.PNG']
  print("path,value")
  for dirpath, dirnames, filenames in os.walk(outdir):
    for filename in filenames:
      n = config.label_names[re.sub(r'[0-9]{1,3}\.(png)$', "", filename)]
      (fn,ext) = os.path.splitext(filename)
      if ext.upper() in exts:
        img_path = os.path.join(dirpath, filename)
        print ('%s,%s' % (img_path, n))
    for dirname in dirnames:
      if dirname in config.label_names:
        n = config.label_names[dirname]
        member_dir = os.path.join(dirpath, dirname)
        for dirpath2, dirnames2, filenames2 in os.walk(member_dir):
          if not dirpath2.endswith(dirname):
            continue
          for filename2 in filenames2:
            (fn,ext) = os.path.splitext(filename2)
            if ext.upper() in exts:
              img_path = os.path.join(dirpath2, filename2)
              print ('%s,%s' % (img_path, n))
