import cv2
import numpy as np


img = cv2.imread('bad_img.jpg')
#Function used for white balance
def white_balance(img):
    #change color space used to represent the colors from rgb to L.A.b 
    # FOR MORE INFO ON LAB:https://en.wikipedia.org/wiki/CIELAB_color_space
    #LAB is useful because it'll change the color without the brightness
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
    avg_a = np.average(result[:, :, 1])#average all the a scale in the LAB space
    avg_b = np.average(result[:, :, 2])#average all the a scale in the LAB space
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) )#substract the average from the a to blance the color
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) )#substract the average from the b to blance the color
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)#convert back to the rgb color space
    return result
#Function used for decrease brightness
def color_change(img):
    import random
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) 
    result[:, :, 1] = result[:, :, 1] +((-1)**(random.randint(1,2)))* random.randint(0,20) #Add random integer to color
    result[:, :, 2] = result[:, :, 2] +((-1)**(random.randint(1,2)))* random.randint(0,20) #Add random integer to color
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)#convert back to the rgb color space
    return result



final2 =  white_balance(img)
print('display')

cv2.imwrite('change.jpg', final2)


###
# The CIELAB space is three-dimensional, and covers the entire range 
# of human color perception, or gamut. It is based on the opponent color 
# model of human vision, where red and green form an opponent pair, and 
# blue and yellow form an opponent pair. The lightness value, L*, also 
# referred to as "Lstar," defines black at 0 and white at 100. The a* axis 
# is relative to the green–red opponent colors, with negative values toward 
# green and positive values toward red. The b* axis represents the blue–yellow 
# opponents, with negative numbers toward blue and positive toward yellow.

# in cv2:
# L: L / 100 * 255
# A: A + 128
# B: B + 128