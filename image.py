from PIL import Image
import numpy as np


# width(col): 320
#height(height): 243
def image(file):
    #np.set_printoptions(threshold=np.nan)
    im = Image.open(file)
    (width,height) = im.size
    print width
    print height
    greyscale_map = list(im.getdata())
    greyscale_map = np.array(greyscale_map)
    greyscale_map = greyscale_map.reshape((height,width))

    crop_rectangle = (0,160,320,200) #left, up, right, down
    cropped_im = im.crop(crop_rectangle)
    cropped_im.show()
    return greyscale_map


def lipCornerDepressor(image):
    #region: 0,160,320,200
    row = len(image)
    col = len(image[0])
    print "row", row, "col", col
    print image[160]

#eyebrows region: 0,100,320,130

file = "C:/Users/admin/Desktop/yalefaces/yalefaces/subject01.happy"
g_map = image(file)
lipCornerDepressor(g_map)
