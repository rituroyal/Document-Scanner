import cv2
from processing import preprocess_image, get_contours, warp_perspective, auto_crop, enhance_document


biggest = get_contours(edges, img)