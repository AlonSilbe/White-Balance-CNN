import cv2
import os
import random
import matplotlib.pyplot as plt



def one_color_change(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
    result[:, :, 1] = result[:, :, 1] +((-1)**(random.randint(1,2)))* random.randint(0,20) #Add random integer to color
    result[:, :, 2] = result[:, :, 2] +((-1)**(random.randint(1,2)))* random.randint(0,20) #Add random integer to color
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)#convert back to the rgb color space
    return result
img=cv2.imread("img.jpg")
img_changed=one_color_change(img)
cv2.imwrite("bad_img.jpg",img_changed)
