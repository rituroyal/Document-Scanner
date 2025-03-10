import cv2
import numpy as np

def preprocess_image(image_path):
    """ Reads and preprocesses the image for contour detection. """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (960, 1280))  # Resizing for consistency
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, 190, 190)
    return img, gray, blurred, edges

def auto_crop(img):
    """ Automatically crops the document by removing white/black borders. """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # If no contours found, return original image

    # Get the largest contour (assumed to be the document)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Crop the image to the bounding box of the document
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img