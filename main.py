import cv2
from processing import preprocess_image, get_contours, warp_perspective, auto_crop, enhance_document

# Load and process image
image_path = r"C:\Users\pragy\OneDrive\Desktop\Document Scanner 2\Image\image.jpg"
img, _, _, edges = preprocess_image(image_path)#Muskan...
biggest = get_contours(edges, img)#Pragya.....
