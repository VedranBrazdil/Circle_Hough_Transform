import cv2
import numpy as np


#GIVE A PATH OF IMAGE YOU WANT TO BE USED
path = 'IMG_NAME.jpg'
img = cv2.imread(path)


#BLUR ---1 for yes, 0 for no blur
bl = 0

#redivi = height, stupci = width
redovi = img.shape[0]
stupci = img.shape[1]


"""---FUNCTIONS-----"""

#showing the picture in original form
def showimg(img):
    cv2.imshow('naslov', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#grayscale
def grayscalef(img):
    #new array
    imggray = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
    r = 0
    g = 0
    b = 0

    redovi = img.shape[0]
    stupci = img.shape[1]
    for x in xrange(redovi):
        for y in xrange(stupci):
            #first version of formula for grayscale
            r = img[x, y][0] * 0.299
            g = img[x, y][1] * 0.587
            b = img[x, y][2] * 0.114

            #second one
            """r = img[x, y][0] * 0.2126
            g = img[x, y][1] * 0.7152
            b = img[x, y][2] * 0.0722"""

            imggray[x,y] = r + g + b
    return imggray


#Gaussian BLUR
def blurf(image, kernel):
    image = image.astype(np.float32)
    output = np.zeros( (image.shape[0] -kernel.shape[0] + 3,
                        image.shape[1] - kernel.shape[1]+ 3))
    kernel_rev = kernel[::-1,::-1]


    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            try:
                for k in range(kernel.shape[0]):
                    for l in range(kernel.shape[1]):
                        output[i,j] += image[i+k,j+l] * kernel_rev[k,l]
            except:
                continue

    #change all numbers into INT:
    #and we keep the size of original image:
    for i in xrange(output.shape[0]):
        for j in xrange(output.shape[1]):
            output[i,j] = int(output[i,j])
            if i == output.shape[0] or i == output.shape[0] - 1 or i == output.shape[0] - 2:
                output[i, j] = ((image[i, j] + image[i, j-1])/2)
            if j == output.shape[1] or j == output.shape[1] - 1 or j == output.shape[1] - 2:
                output[i, j] = ((image[i, j] + image[i-1, j])/2)


    output[output>255] = 255
    output[output<0]   = 0
    output = output.astype(np.uint8)
    return output


#edge detection
def edgef(image, kernel):
    output = np.zeros( (image.shape[0] -kernel.shape[0] + 3,
                        image.shape[1] - kernel.shape[1]+ 3))
    kernel_rev = kernel[::-1,::-1]

    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            try:
                for k in range(kernel.shape[0]):
                    for l in range(kernel.shape[1]):
                        output[i,j] += image[i+k,j+l] * kernel_rev[k,l]
            except:
                continue

    #we keep the size of original image:
    for i in xrange(output.shape[0]):
        for j in xrange(output.shape[1]):
            if i == output.shape[0] or i == output.shape[0] - 1 or i == output.shape[0] - 2:
                output[i,j] = 0
            if j == output.shape[1] or j == output.shape[1] - 1 or j == output.shape[1] - 2:
                output[i,j] = 0

    output[output>255] = 255
    output[output<0]   = 0
    output = output.astype(np.uint8)
    return output




#circle Hough Transformation
def hough(img, path, bl):

    #deep copy - we make new picture and we draw circles on it
    imgnew = cv2.imread(path)

    """Parameters:"""
    #min radius and max radius of circles we are looking for
    rmin = 16
    rmax = 25

    #minimum threshold of voting to be considerated as center of circle with radius r
    minH = 230
    #220 avg

    #creating the sin and cos of 360:
    sinang = dict()
    cosang = dict()
    for angle in range(0, 360):
        sinang[angle] = np.sin(angle * np.pi / 180)
        cosang[angle] = np.cos(angle * np.pi / 180)

    H = np.zeros((img.shape[0],img.shape[1], (rmax+1)), dtype=np.uint32)

    #voting function
    for x in range(0, img.shape[0]):
        print ("row: " + str(x)) #THIS IS ROW CHECKER SO WE KNOW IT DID NOT GET BUGED
        for y in range(0, img.shape[1]):
            if img[x, y] == 255:
                for r in range(rmin, rmax):
                    for angle in range(0, 360):
                       b = y - round(r * sinang[angle])
                       a = x - round(r * cosang[angle])
                       if a >= 0 and a < img.shape[0] and b >= 0 and b < img.shape[1]:
                           H[a, b, r] = H[a, b, r] + 1




    """CIRCLE DRAWING"""
    max = 0
    a1 = 0
    b1 = 0

    for r in range(rmin, rmax):
        for a in range(0, img.shape[0]):
            for b in range(0, img.shape[1]):
                if H[a, b, r] > minH:
                    #we look for local maximum in matrix of 4x3
                    #this part could be better if you also look for loxal maximum in 3D, but this one also gives good results
                    max = H[a, b, r] #A,B
                    H[a, b, r] = 0
                    a1 = a
                    b1 = b
                    if a+2 <= img.shape[0]:
                        if b+2 <= img.shape[1]:
                            if H[a+1, b-1, r] > max: #A+1, B-1
                                max = H[a + 1, b - 1, r]
                                a1 = a+1
                                b1 = b-1
                            H[a + 1, b - 1, r] = 0
                            if H[a+2, b-1, r] > max: #A+2, B-1
                                max = H[a + 2, b - 1, r]
                                a1 = a+2
                                b1 = b-1
                            H[a + 2, b - 1, r] = 0

                            for k in range(0, 2):
                                for l in range (0, 2):
                                    if H[a + k, b + l, r] >= max:
                                        max = H[a + k, b + l, r]
                                        a1 = a + k
                                        b1 = b + l
                                    H[a + k, b + l, r] = 0
                    #DRAWING OF CIRCLE WITH A1 B1 AS CENTER PARAMETERS
                    cv2.circle(imgnew, (b1, a1), r, color=(0, 0, 255), thickness=1, lineType=8, shift=0)



    #END OF FUNCTION- SAVING AN IMAGE
    cv2.imwrite("Seminar/HUGHv2 - r" + str(rmin) + "-" + str(rmax) + " - " + str(minH) + " blur" + str(bl) + ".bmp", imgnew)
    return imgnew





""""""
""""""
""""MAIN PROGRAM"""
#original image
showimg(img)


#every picture will be saved in a folder names "Seminar", so make one and name it the same <---------------DONT FORGET THIS-------------------

#GRAY
IMG_GRAY = grayscalef(img)

showimg(IMG_GRAY)
print (IMG_GRAY.shape)
cv2.imwrite("Seminar/GRAY.bmp", IMG_GRAY)



#here we check if blur is needed or not
if bl == 1:
    #Gaussian blur 3x3
    kernelb = ([1,2,1],[2,4,2],[1,2,1])
    kernelb = np.array(kernelb)
    kernelb = kernelb.astype(np.float32)
    kernelb = (kernelb/16)
    IMG_BLUR = blurf(IMG_GRAY, kernelb)
else:
    #no blura
    IMG_BLUR = IMG_GRAY

showimg(IMG_BLUR)
print (IMG_BLUR.shape)
cv2.imwrite("Seminar/BLUR.bmp", IMG_BLUR)



#EDGE
kernele = ([-1,-1,-1],[-1,8,-1],[-1,-1,-1])
kernele = np.array(kernele)
IMG_EDGE = edgef(IMG_BLUR, kernele)

#binarization
IMG_EDGE[IMG_EDGE < 75] = 0
IMG_EDGE[IMG_EDGE > 74] = 255

showimg(IMG_EDGE)
print (IMG_EDGE.shape)
cv2.imwrite("Seminar/EDGE.bmp", IMG_EDGE)



#circle Hough Transform
IMG_HOUGH = hough(IMG_EDGE, path, bl)
showimg(IMG_HOUGH)
print (IMG_HOUGH.shape)
