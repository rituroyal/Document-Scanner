import cv2
from processing import preprocess_image, get_contours, warp_perspective, auto_crop, enhance_document

# Load and process image
image_path = r"C:\Users\Asus\OneDrive\Desktop\Scanify\image\image.jpg"
img, _, _, edges = preprocess_image(image_path)
biggest = get_contours(edges, img)
# Warp and crop
warped_img = warp_perspective(img, biggest)
