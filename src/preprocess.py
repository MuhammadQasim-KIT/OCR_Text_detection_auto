import cv2
import numpy as np

# ----------------------------
# Helpers
# ----------------------------

def _rotate_image_keep_bounds(img: np.ndarray, angle_deg: float, bg: int = 255) -> np.ndarray:
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=bg)


def deskew_binary(binary: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(binary < 128))
    if coords.size < 200:
        return binary

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return binary

    return _rotate_image_keep_bounds(binary, angle_deg=angle, bg=255)


def remove_table_lines(binary: np.ndarray) -> np.ndarray:
    inv = 255 - binary

    h, w = inv.shape[:2]
    horiz_len = max(20, w // 25)
    vert_len = max(20, h // 25)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

    horiz = cv2.erode(inv, horiz_kernel, iterations=1)
    horiz = cv2.dilate(horiz, horiz_kernel, iterations=1)

    vert = cv2.erode(inv, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, vert_kernel, iterations=1)

    lines = cv2.bitwise_or(horiz, vert)
    no_lines = cv2.subtract(inv, lines)
    out = 255 - no_lines

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
    return out


# ----------------------------
# Preprocessing Variants
# ----------------------------

def preprocess_adaptive(bgr_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    if np.mean(thr) < 127:
        thr = 255 - thr

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    return thr


def preprocess_adaptive_deskew(bgr_img: np.ndarray) -> np.ndarray:
    return deskew_binary(preprocess_adaptive(bgr_img))


def preprocess_otsu(bgr_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(thr) < 127:
        thr = 255 - thr

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thr = cv2.dilate(thr, kernel, iterations=1)
    return thr


def preprocess_otsu_deskew(bgr_img: np.ndarray) -> np.ndarray:
    return deskew_binary(preprocess_otsu(bgr_img))


def preprocess_otsu_table(bgr_img: np.ndarray) -> np.ndarray:
    return remove_table_lines(preprocess_otsu(bgr_img))


def preprocess_otsu_table_deskew(bgr_img: np.ndarray) -> np.ndarray:
    thr = preprocess_otsu(bgr_img)
    thr = deskew_binary(thr)
    thr = remove_table_lines(thr)
    return thr


def preprocess_for_ocr(bgr_img: np.ndarray) -> np.ndarray:
    return preprocess_adaptive(bgr_img)
