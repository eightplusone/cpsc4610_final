from PIL import Image
import numpy as np

# Convert an image to a greyscale map (2D array of integer value
# in the range of 0-255)
def image2greyscaleMap(file):
	im = Image.open(file)

	# Get size
	(width, height) = im.size
	# Get data from the loaded file
	greyscale_map = list(im.getdata())
	# Convert the data into a 2D array
	greyscale_map = np.array(greyscale_map)
	greyscale_map = greyscale_map.reshape((height, width))

	return greyscale_map


# Test
#file = "yalefaces/subject01.happy"
#g_map = image2greyscaleMap(file)
#print(g_map)
