'''
Created on Jan 13, 2019

@author: AMG
'''
import cv2
import numpy as np
import rawpy
#https://letmaik.github.io/rawpy/api/rawpy.RawPy.html
#https://letmaik.github.io/rawpy/api/rawpy.Params.html#rawpy.Params
import os
import matplotlib.pyplot as plt
from cv2 import cvtColor


class RawImRead:
    
    raw_im = None
    
    def __init__(self, lib_path=None, is_raw=False,is_display=False):
        self.lib_path = lib_path
        self.is_raw = is_raw    # currently not in use
        self.is_display = is_display # if True the read image is displayed 
        
    
    
    def imread(self, lib_path=None, im_name=None): 
        im_path = ''
        if lib_path is None and self.lib_path is None:
            print ('Warning - image is read from projects root lib')
        if lib_path is None:
            im_path = os.path.join(self.lib_path, im_name)
        else:
            im_path = os.path.join(self.lib_path, im_name)
        
        
        if not os.path.isfile(im_path):
            print ('Error - invalid filename or folder')
            return False, np.array([])
        
        filename, file_extension = os.path.splitext(im_path)
        # we are handling different file extensions with different calls 
        # if opencv can't read the file format we'll use different command 
        if file_extension == '.dng':
            with rawpy.imread(im_path) as raw:
                self.raw_im = raw.postprocess(four_color_rgb = True)
        else:
            self.raw_im = cv2.imread(im_path, cv2.IMREAD_ANYCOLOR)
            self.raw_im = cv2.cvtColor(self.raw_im, cv2.COLOR_BGR2RGB)
        
        if self.is_display is True:
            plt.imshow(self.raw_im)
            plt.show()
        
        return True, self.raw_im.astype(float) # we are casting file format to float (for firther processing
            