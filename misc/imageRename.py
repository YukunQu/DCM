# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 14:17:13 2021

@author: qyk
"""

import os


# rename all the images 
imgdir = r'C:\Myfile\File\工作\PhD\Development cognitive map\material\被选的30张图片'
img_list = os.listdir(imgdir)
for i,img_name in enumerate(img_list):
    img_ori_name = os.path.join(imgdir,img_name)
    img_new_name = os.path.join(imgdir,'{}.png'.format(i+1))
    os.rename(img_ori_name,img_new_name)