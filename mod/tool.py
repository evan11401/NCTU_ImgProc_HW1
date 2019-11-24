import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
plt.ion()

def calculateHist(image):
    y = np.bincount(image.ravel(), minlength=256)
    return y

def showHist(image,col):
    color = ('b','g','r')
    color2 = ('cyan','lime','pink')
    useColor = color if col==1 else color2
    for i, col in enumerate(useColor):
        histr = calculateHist(image.T[i])
        plt.plot(histr, color = col)
        plt.xlim([0, 256])
    plt.show()    
    
def gamma(image):
    new_image = np.zeros(image.shape, image.dtype)
    gamma = 1.0 # Simple contrast control
    # Initialize values
    print(' Gamma Correction ')
    print('-------------------------')
    try:
        gamma = float(input('* Enter the gamma value Gamma < 1 ~ Dark ; Gamma > 1 ~ Bright: ')) #1.0
    except ValueError:
        print('Error, not a number')

    # Gamma < 1 ~ Dark ; Gamma > 1 ~ Bright
    new_image = 255*((image/255.0)**(1/gamma))

    new_image = new_image.astype('uint8')
    return new_image

def linear(image):
    print(' Basic Linear Transforms ')
    print('-------------------------')
    new_image = np.zeros(image.shape, image.dtype)
    alpha = 1.0 # Simple contrast control
    beta = 0    # Simple brightness control
    try:
        alpha = float(input('* Enter the alpha value [1.0-3.0] contrast control: '))
        beta = int(input('* Enter the beta value [0-100] brightness control: ')) 
    except ValueError:
        print('Error, not a number')
    # for y in range(image.shape[0]):
    #     for x in range(image.shape[1]):
    #         for c in range(image.shape[2]):
    #             new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    new_image = np.clip(image*alpha + beta,0,255)
    new_image = new_image.astype('uint8')
    return new_image
    
def HistEqual(image):
    (b, g, r) = cv.split(image)
    bH = cv.equalizeHist(b)
    gH = cv.equalizeHist(g)
    rH = cv.equalizeHist(r)
    new_image = cv.merge((bH, gH, rH))
    return new_image

def oneChannel(image):
    gamma = 1.0 # Simple contrast control
    # Initialize values
    print(' Gamma Correction one Channel')
    print('-------------------------')
    try:
        gamma1 = float(input('* Enter the b gamma value Gamma < 1 ~ Dark ; Gamma > 1 ~ Bright: '))
        gamma2 = float(input('* Enter the g gamma value Gamma < 1 ~ Dark ; Gamma > 1 ~ Bright: ')) 
        gamma3 = float(input('* Enter the r gamma value Gamma < 1 ~ Dark ; Gamma > 1 ~ Bright: ')) 
    except ValueError:
        print('Error, not a number')
    (b, g, r) = cv.split(image)
    bH = 255*((b/255.0)**(1/gamma1))
    gH = 255*((g/255.0)**(1/gamma2))
    rH = 255*((r/255.0)**(1/gamma3))
    new_image = cv.merge((bH, gH, rH))
    new_image = new_image.astype('uint8')
    
    return new_image

def bilateralFilter(image):
    new_image = cv.bilateralFilter(image,7,75,75)
    return new_image

def medianBlur(image):
    return cv.medianBlur(image,3)
def noise(image):
    return cv.fastNlMeansDenoisingColored(image)

def hsvAdjust(image):
    print(' hsvAdjust')
    print('-------------------------')
    hv = 0
    sv = 0
    vv = 0
    try:
        hv = int(input('* Enter the Hue value: '))
        sv = int(input('* Enter the Saturation value: ')) 
        vv = int(input('* Enter the Lightness value: ')) 
    except ValueError:
        print('Error, not a number')
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    if hv<0:
        h -= -hv
    else:
        h += hv # Hue
    if sv<0:
        s -= -sv
    else:
        s += sv # Saturation
    if vv<0:
        v -= -vv
    else:
        v += vv # Lightness
    final_hsv = cv.merge((h, s, v))
    final_rgb = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return np.clip(final_rgb, 0, 255)

class Channel_value:
    val = -1.0
    intensity = -1.0


def find_intensity_of_atmospheric_light(img, gray):
    top_num = int(img.shape[0] * img.shape[1] * 0.001)
    toplist = [Channel_value()] * top_num
    dark_channel = find_dark_channel(img)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            val = img.item(y, x, dark_channel)
            intensity = gray.item(y, x)
            for t in toplist:
                if t.val < val or (t.val == val and t.intensity < intensity):
                    t.val = val
                    t.intensity = intensity
                    break

    max_channel = Channel_value()
    for t in toplist:
        if t.intensity > max_channel.intensity:
            max_channel = t

    return max_channel.intensity


def find_dark_channel(img):
    return np.unravel_index(np.argmin(img), img.shape)[2]


def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))


def dehaze(img, windowSize):
    w = 0.95
    t0 = 0.55
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    light_intensity = find_intensity_of_atmospheric_light(img, gray)
    size = (img.shape[0], img.shape[1])

    outimg = np.zeros(img.shape, img.dtype)

    for y in range(size[0]):
        for x in range(size[1]):
            x_low = max(x-(windowSize//2), 0)
            y_low = max(y-(windowSize//2), 0)
            x_high = min(x+(windowSize//2), size[1])
            y_high = min(y+(windowSize//2), size[0])

            sliceimg = img[y_low:y_high, x_low:x_high]

            dark_channel = find_dark_channel(sliceimg)
            t = 1.0 - (w * img.item(y, x, dark_channel) / light_intensity)

            outimg.itemset((y,x,0), clamp(0, ((img.item(y,x,0) - light_intensity) / max(t, t0) + light_intensity), 255))
            outimg.itemset((y,x,1), clamp(0, ((img.item(y,x,1) - light_intensity) / max(t, t0) + light_intensity), 255))
            outimg.itemset((y,x,2), clamp(0, ((img.item(y,x,2) - light_intensity) / max(t, t0) + light_intensity), 255))
    return outimg

def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return  np.exp(- (x ** 2) / (2 * sigma ** 2))

def bilateralfilter(image, filtered_image, x, y, dia, sigma_color, sigma_distance):
    hl = dia/2
    i_filtered = 0
    Wp = 0
    i = 0
    for i in range(dia):
        for j in range(dia):
            near_x = int(np.mod(x - (hl - i), len(image)))
            near_y = int(np.mod(y - (hl - j), len(image[0])))
            ws = gaussian(image[near_x][near_y] - image[x][y], sigma_color)
            wr = gaussian(distance(near_x, near_y, x, y), sigma_distance)
            w = ws * wr
            i_filtered += image[near_x][near_y] * w
            Wp += w
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = i_filtered

def bilateral_filter(image, filter_dia, sigma_color, sigma_distance):
    filtered_image = np.zeros(image.shape, image.dtype)
    i = 0
    while i < len(image):
        j = 0
        while j < len(image[0]):
            bilateralfilter(image, filtered_image, i, j, filter_dia, sigma_color, sigma_distance)
            j += 1
        i += 1
    return filtered_image