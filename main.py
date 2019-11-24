from __future__ import print_function
from builtins import input
import cv2 as cv
import numpy as np
import argparse
import mod.tool as tool
import matplotlib.pyplot as plt
plt.ion()
# Read image given by user
parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! tutorial.')
parser.add_argument('--input', help='Path to input image.', default='p1im1.bmp')
args = parser.parse_args()
# filename = 'p1im'+args.input+'.bmp'
filename = args.input
# outputname = 'P1im'+args.input+'_0856043.bmp'
outputname = args.input + 'output.bmp'
print(filename)
image = cv.imread(cv.samples.findFile(filename))

if image is None:
    print('Could not open or find the image: ', args.input)
    exit(0)

new_image = image

#adjust the image
# new_image = tool.gamma(new_image)
# new_image = tool.dehaze(new_image,20)
# new_image = tool.bilateral_filter(new_image, 7, 75, 75)
new_image = tool.dehaze(new_image,5)

#show the image before and after
cv.imshow('Original Image', image)
cv.imshow('New Image', new_image)

#show the histogram
tool.showHist(image,1)
tool.showHist(new_image,2)

#write the image
cv.imwrite(outputname, new_image)

plt.ioff()
plt.show()
