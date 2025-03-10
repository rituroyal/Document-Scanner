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

def warp_perspective(img, biggest, width=480, height=640):
    """ Applies a perspective transform and auto-crops the document. """
    if biggest is None:
        return img  # If no document is detected, return original image

    ordered_pts = order_points(biggest)
    pts1 = np.float32(ordered_pts)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (width, height))

    # Apply auto-cropping after warping
    return auto_crop(img_warp)

def get_contours(edges, img):
    """ Finds the largest quadrilateral contour, assuming it's the document. """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, max_area = [], 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                biggest, max_area = approx, area

    return biggest.reshape(4, 2) if len(biggest) == 4 else None

def order_points(pts):
    """ Orders contour points: [Top-Left, Top-Right, Bottom-Left, Bottom-Right] """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[3] = pts[np.argmax(s)]  # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[2] = pts[np.argmax(diff)]  # Bottom-left
    return rect
