'''
Created on 20 Jun 2021

@author: danan
'''
import cv2
import strel_func as stf
import numpy as np
import matplotlib.pyplot as plt


def Ellipse_Correction(im_orig_copy, cy,cx):
    
    orig_im_grad_x= cv2.Sobel(im_orig_copy.astype('float64'),cv2.CV_64F,0,1,ksize=7)
    orig_im_grad_y= cv2.Sobel(im_orig_copy.astype('float64'),cv2.CV_64F,1,0,ksize=7)
    orig_im_grad = np.sqrt(np.power(orig_im_grad_x,2)+np.power(orig_im_grad_y,2))
    
    ONH_Center = ((cy.max()-cy.min())//2+cy.min(),(cx.max()-cx.min())//2+cx.min())
    
    cx_points = cx[::4]
    cy_points = cy[::4]

    
    for m in range(len(cx_points)-1):
        '''
        # Nearest points
        cx_diff = cx_points-cx_points[m]
        cy_diff = cy_points-cy_points[m]
        
        distace_to_pixel = np.sqrt(cx_diff**2 +cy_diff**2)
        distace_to_pixel[m]= 1000
        m_1 = np.argmin(distace_to_pixel)
        distace_to_pixel[m_1]= 1000
        m_2 = np.argmin(distace_to_pixel)
        '''
        vec_angle = np.rad2deg(np.arctan2(cx_points[m] - ONH_Center[1], cy_points[m] - ONH_Center[0]))
        
        strel_line = stf.strel(50,vec_angle+90) 
        
        line_im = np.zeros(im_orig_copy.shape)
        
        try:
            line_im[cy_points[m]-strel_line.shape[0]//2:cy_points[m]+strel_line.shape[0]//2+1, cx_points[m]-strel_line.shape[1]//2:cx_points[m]+strel_line.shape[1]//2+1]= strel_line
        except:
            line_im = np.zeros(im_orig_copy.shape)
            
        max_arg = np.unravel_index(np.argmax(np.multiply(line_im,orig_im_grad)),line_im.shape)
        max_value = orig_im_grad[max_arg[0],max_arg[1]]
        current_value = orig_im_grad[cy_points[m],cx_points[m]]
        
        if max_value > current_value*1.2:
            cy_points[m] = max_arg[0]
            cx_points[m] = max_arg[1]
        
        im_orig_copy2 = im_orig_copy.copy()
        im_orig_copy2 = np.dstack((im_orig_copy2,im_orig_copy2,im_orig_copy2))
        
        im_orig_copy2[line_im!=0] = (255,0,0)
      
        im_orig_copy2[cy_points[m],cx_points[m]]=(0,0,255)
        #im_orig_copy2[cy_points[m_1],cx_points[m_1]]=(0,0,255)
        #im_orig_copy2[cy_points[m_2],cx_points[m_2]]=(0,0,255)
        '''
        plt.figure(1)
        plt.imshow(im_orig_copy2)
        plt.show()
        '''       
    return cx_points,cy_points
            
        