

import cv2
import numpy as np


def is_photo(img):
    """Guess if the picture is a PHOTO (face, scenery) or a TEXT PAGE.
    A text page is almost only white paper and black ink, so it has
    almost NO colour. A photo has a lot of colour (blue wall, skin...).
    We count how many pixels are 'colourful'."""
    small = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # S = saturation (how strong the colour is), V = brightness
    colourful = (hsv[..., 1] > 60) & (hsv[..., 2] > 50)
    return float(colourful.mean()) > 0.20


def auto_crop(img):
    """Cuts the empty borders around a TEXT page. Ignores tiny specks."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = max(15, w // 12)
    blocks = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(blocks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    boxes = [cv2.boundingRect(c) for c in contours]
    biggest_area = max(bw * bh for _, _, bw, bh in boxes)
    keep = [b for b in boxes if b[2] * b[3] >= 0.10 * biggest_area]
    x1 = min(b[0] for b in keep)
    y1 = min(b[1] for b in keep)
    x2 = max(b[0] + b[2] for b in keep)
    y2 = max(b[1] + b[3] for b in keep)

    if (x2 - x1) * (y2 - y1) < 0.10 * w * h:
        return img
    return img[y1:y2, x1:x2]


def order_points(pts):
    """Orders points: [Top-Left, Top-Right, Bottom-Left, Bottom-Right]"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
    return rect


def warp_perspective(img, biggest, crop=True):
    """Flattens the paper into a straight rectangle."""
    if biggest is None:
        return img

    rect = order_points(biggest)
    tl, tr, bl, br = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return img

    pts1 = np.float32(rect)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (width, height))
    return auto_crop(img_warp) if crop else img_warp


def get_contours(edges, min_area=500):
    """Finds the biggest 4-corner shape. We assume it is the paper."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, max_area = None, 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                biggest, max_area = approx, area

    return biggest.reshape(4, 2) if biggest is not None else None


def draw_contour(img, biggest):
    if biggest is not None:
        cv2.polylines(img, [biggest.astype(np.int32)], True, (0, 255, 0), 3)
    return img


def enhance_text_page(img):
    """For TEXT pages: even light, white paper, dark sharp text (grey output)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    if w < 1600:
        s = 1600.0 / w
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        h, w = gray.shape

    k1 = max(7, (w // 200) | 1)
    k2 = max(21, (w // 60) | 1)
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((k1, k1), np.uint8)), k2)
    norm = cv2.divide(gray, bg, scale=255)

    lut = np.clip((np.arange(256) - 70) * 255.0 / (225 - 70), 0, 255).astype(np.uint8)
    out = cv2.LUT(norm, lut)

    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.6, blur, -0.6, 0)


def enhance_photo(img):
    """For PHOTOS: keep the colours and the faces. Only fix light a little
    and make it a bit sharper (colour output)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE = makes dark and bright parts balanced, in small tiles
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.4, blur, -0.4, 0)


def enhance_photo_gray(img):
    """For PHOTOS in black and white, like a printer does.
    Step 1: change colour to shades of grey (grayscale).
    Step 2: balance the light, so face and shirt are both visible.
    Step 3: sharpen a little."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # colour -> grey shades

    # CLAHE = balances dark and bright parts in small tiles
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    return cv2.addWeighted(gray, 1.4, blur, -0.4, 0)


def process_image(img, mode="auto", output="bw"):
    """Full pipeline: BGR image in -> scanned page out.
    mode   = "auto" (decide by itself), "text" or "photo"
    output = "bw" (black and white, like a printer) or "color" """
    h, w = img.shape[:2]
    scale = 2600.0 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    if mode == "photo":
        photo = True
    elif mode == "text":
        photo = False
    else:
        photo = is_photo(img)

    # 1) find the paper corners on a small copy
    small = cv2.resize(img, (960, 1280))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, 190, 190)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    biggest = get_contours(edges, min_area=0.30 * 960 * 1280)

    # 2) flatten (photos are NOT cropped by ink, so we never cut the face)
    if biggest is not None:
        biggest = biggest.astype(np.float32) * np.array([w / 960.0, h / 1280.0], dtype=np.float32)
        warped = warp_perspective(img, biggest, crop=not photo)
    else:
        warped = img if photo else auto_crop(img)

    # 3) clean it up in the right way
    if not photo:
        return enhance_text_page(warped)          # text: always clean grey/white
    if output == "bw":
        return enhance_photo_gray(warped)         # photo in black and white
    return enhance_photo(warped)                  # photo in colour