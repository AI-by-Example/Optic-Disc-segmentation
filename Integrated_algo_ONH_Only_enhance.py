'''
Created on 20 Jun 2021

@author: danan
'''

import matplotlib.pyplot as plt
from RawImRead import RawImRead as RIR
import Ellipse_Correction as EC
import numpy as np
import os
from scipy.signal import convolve2d
from skimage import morphology
import numpy as np
from numpy import savetxt,loadtxt
from sklearn.svm import LinearSVC
import create_feature_mat_classify as FMC
import create_feature_mat_OC_classify as FMOC

from skimage import io
import cv2
from skimage.measure import  regionprops
from skimage.morphology import disk, closing
from skimage.transform import hough_ellipse
from skimage.feature import canny
from skimage.draw import ellipse_perimeter
from PIL import Image
from skimage.segmentation import flood, flood_fill

from skimage.filters import gaussian
from skimage.segmentation import active_contour, inverse_gaussian_gradient

# train model ONH
feature_mat_norm_all = loadtxt('feature_mat_norm_all.csv', delimiter=',')
Labels_vec_all = loadtxt('Labels_vec_all.csv', delimiter=',')
clf = LinearSVC(C=1, max_iter=10000)
model = clf.fit(feature_mat_norm_all, Labels_vec_all)

'''
# Training Model OC
feature_mat_norm_all_OC = loadtxt('feature_mat_norm_all_OC.csv', delimiter=',')
Labels_vec_all_OC = loadtxt('Labels_vec_all_OC.csv', delimiter=',')
clf_oc = LinearSVC(C=10, max_iter=100000)
model_OC = clf_oc.fit(feature_mat_norm_all_OC, Labels_vec_all_OC)
'''

for i in range(1,21):
    print(i)
    # Import Image
    dir_path = os.path.normpath(r'C:\Users\danan\Downloads\BinRushedcorrected\BinRushed\BinRushed3')
    
    res_dir_path = os.path.join(dir_path, 'res/')
    im_name = 'image' + str(i) + 'prime.jpg'
    
    res_dir_path = res_dir_path.replace(r'/', '\\')
    # init classes
    rir = RIR(lib_path = dir_path, is_display = False)
    
    im = None
    #read image
    ret, img = rir.imread(im_name=im_name)
    
    if len(img.shape) > 2:
        #im = img[500:1300,500:1500,1] # Green Layer
        im = img[:,:,1] # Green Layer
        im_blue = img[:,:,2]
        im_red = img[:,:,0]
        #im = np.divide(((img[:,:,1] + img[:,:,2])/2),(img[:,:,0] + 1))
    else:
        im = img.copy()
     
    im_orig = io.imread(r'C:\Users\danan\Downloads\BinRushedcorrected\BinRushed\BinRushed3\\' + im_name)
    im_orig_large=im_orig.copy()
    im_large =  im.copy()
    im_blue_large = im_blue.copy()
    im_red_large = im_red.copy()
    
    
    # Localization
    
    im_copy=im.copy()
    
    Threshold = 20
    if (im > Threshold).sum()/im.size < 0.01:
        while  (im>Threshold).sum()/im.size < 0.01:
            Threshold = Threshold-1
    else:
        while  (im>Threshold).sum()/im.size > 0.01:
            Threshold = Threshold+1
    
    Binary_thresh = im > Threshold
    
    ret, labels = cv2.connectedComponents(Binary_thresh.astype(np.uint8))
    regions = regionprops(labels)
    
    for region in regions:
        #print(region.label)
        #print(region.area)
        if region.area<2000 :
            labels[labels==region.label] = 0
        elif region.major_axis_length/region.minor_axis_length > 5:    
            im_copy = flood_fill(im_copy, (int(region.centroid[0]), int(region.centroid[1])), 0, tolerance=40) 
            labels[labels==region.label] = 0
    
    Threshold = 20
    if (im_copy > Threshold).sum()/im_copy.size < 0.01:
        while  (im_copy>Threshold).sum()/im_copy.size < 0.01:
            Threshold = Threshold-1
    else:
        while  (im_copy>Threshold).sum()/im_copy.size > 0.01:
            Threshold = Threshold+1        
    
    Binary_thresh = im_copy > Threshold 
    
    ret, labels = cv2.connectedComponents(Binary_thresh.astype(np.uint8))
    regions = regionprops(labels)
    
    region_value_max = 0
    region_area_max_label = 0
    for region in regions:
        
        if region.area<2000 :
            labels[labels==region.label] = 0 
        
        elif np.max(im_copy[labels==region.label])>region_value_max:
            region_value_max = np.max(im_copy[labels==region.label])
            region_area_max_label = region.label
    
    max_ind = np.zeros((2,1))    
    max_ind_0 = int(regions[region_area_max_label-1].centroid[0]) 
    max_ind_1 = int(regions[region_area_max_label-1].centroid[1])
    
    
    im_orig = im_orig[max_ind_0-200:max_ind_0+200, max_ind_1-200:max_ind_1+200,:]
    im = im[max_ind_0-200:max_ind_0+200, max_ind_1-200:max_ind_1+200]
    im_red = im_red[max_ind_0-200:max_ind_0+200, max_ind_1-200:max_ind_1+200]
    im_blue = im_blue[max_ind_0-200:max_ind_0+200, max_ind_1-200:max_ind_1+200]
    '''
    plt.figure(1)
    plt.imshow(im_orig, cmap='gray')
    
    
    
    plt.show()
    '''
    # ONH Algorithm
    '''
    feature_mat_norm_all = loadtxt('feature_mat_norm_all.csv', delimiter=',')
    Labels_vec_all = loadtxt('Labels_vec_all.csv', delimiter=',')
    
    clf = LinearSVC(C=1, max_iter=10000)
    model = clf.fit(feature_mat_norm_all, Labels_vec_all)
    '''
    numSegments = 200
    
    feature_mat_norm_test, segments = FMC.create_feature_mat_classify(im_orig, im_red, im, im_blue, numSegments)
    decision_results = model.decision_function(feature_mat_norm_test)
    
    segments2=segments.copy().astype('float')
    for i in range(numSegments):
        segments2[segments==i] = decision_results[i]
    segment_result1 = cv2.blur(segments2,(9,9))
    
    bin_result = segment_result1>np.mean(segment_result1)*0.25
    
    ret, labels = cv2.connectedComponents(bin_result.astype(np.uint8))
    regions = regionprops(labels)
    
    max_region_area = 0
    for region in regions:
        if region.area>max_region_area :
            max_region_label = region.label
            max_region_area=region.area
    
          
    labels = labels==max_region_label
    
    selem = disk(27)
    label_close = closing(labels+0, selem=selem)
    edges = canny(label_close, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
    
    result = hough_ellipse(edges, accuracy=30, threshold=30, min_size=80, max_size=200)      
    result.sort(order='accumulator')
    
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]
    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    
    im_orig_copy = im_orig.copy()
    
    # Below code convert image gradient in x direction
    im_orig_copy= gaussian(im_orig_copy[:,:,0],3)
    orig_im_grad_x= cv2.Sobel(im_orig_copy.astype('float64'),cv2.CV_64F,0,1,ksize=7)
    orig_im_grad_y= cv2.Sobel(im_orig_copy.astype('float64'),cv2.CV_64F,1,0,ksize=7)
    orig_im_grad = np.sqrt(np.power(orig_im_grad_x,2)+np.power(orig_im_grad_y,2))

    
    im_orig[cy, cx] = (0, 0, 255)
    '''
    init = np.array([cx, cy]).T
    
    snake = active_contour(gaussian(im_orig_copy[:,:,1],3),
                       init, alpha=0.0001, beta=1, gamma=0.001,w_edge =10,w_line = 0, max_px_move =2)
    
    snake = snake.astype(int)
    im_orig[snake[:, 1],snake[:, 0]] =(0,255,0)
    '''
    
    cx_real_axes = cx + max_ind_1-200
    cy_real_axes = cy + max_ind_0-200
    
    im_orig_large_marked = im_orig_large.copy()
    im_orig_large_marked[cy_real_axes, cx_real_axes] = (0, 0, 255)
    
    cx_points, cy_points = EC.Ellipse_Correction(im_orig_copy, cy,cx)
    im_orig[cy_points, cx_points] = (255, 0, 0)
    
    new_ellipse = np.zeros(orig_im_grad.shape)
    new_ellipse[cy_points,cx_points] = 255
    
    result = hough_ellipse(new_ellipse, accuracy=30, threshold=30, min_size=80, max_size=200)      
    result.sort(order='accumulator')
    
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]
    # Draw the ellipse on the original image
    cy_new, cx_new = ellipse_perimeter(yc, xc, a, b, orientation)
    
    im_orig[cy_new, cx_new] = (0, 255, 0)
    
    plt.figure(1)
    plt.imshow(segments2)
    
    plt.figure(2)
    plt.imshow(segment_result1)
    
    
    plt.figure(3)
    plt.imshow(bin_result)
    
    plt.figure(3)
    plt.imshow(edges)
    
    plt.figure(4)
    plt.imshow(im_orig)
    
    plt.figure(5)
    plt.imshow(np.abs(orig_im_grad))
    
    plt.figure(6)
    plt.imshow(np.abs(orig_im_grad_x))
    
    plt.figure(7)
    plt.imshow(np.abs(orig_im_grad_y))
    
    plt.figure(8)
    plt.imshow(im_orig_copy)
    
    plt.show()
    
    
    

print('done')