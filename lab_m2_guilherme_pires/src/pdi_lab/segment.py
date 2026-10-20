import cv2
import numpy as np

from pdi_lab.errors import LabError

METHODS = ("fixed", "otsu", "adaptive", "hsv")
HSV_LIMITS = (179, 255, 255)


def to_gray(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def preprocess(image, blur):
    if blur == 0:
        return image
    return cv2.GaussianBlur(image, (blur, blur), 0)


def parse_hsv(text, name):
    try:
        values = [int(v) for v in str(text).split(",")]
    except ValueError:
        raise LabError(f"{name} deve ter o formato H,S,V com inteiros: {text}")
    if len(values) != 3:
        raise LabError(f"{name} deve ter 3 valores H,S,V: {text}")
    for value, limit, channel in zip(values, HSV_LIMITS, "HSV"):
        if not 0 <= value <= limit:
            raise LabError(f"{name}: canal {channel}={value} fora da faixa 0..{limit}")
    return tuple(values)


def threshold_fixed(gray, threshold):
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


def threshold_otsu(gray):
    value, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask, float(value)


def threshold_adaptive(gray, block_size, c):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)


def range_hsv(image_bgr, lower, upper):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, np.array(lower), np.array(upper))
    low_part = cv2.inRange(hsv, np.array((lower[0], lower[1], lower[2])), np.array((179, upper[1], upper[2])))
    high_part = cv2.inRange(hsv, np.array((0, lower[1], lower[2])), np.array((upper[0], upper[1], upper[2])))
    return cv2.bitwise_or(low_part, high_part)


def validate_params(method, threshold=None, block_size=None, c=None, hsv_lower=None, hsv_upper=None, blur=0):
    if method not in METHODS:
        raise LabError(f"metodo desconhecido: {method} (use {', '.join(METHODS)})")
    if threshold is not None and not 0 <= int(threshold) <= 255:
        raise LabError(f"--threshold fora da faixa 0..255: {threshold}")
    if blur != 0 and (blur < 3 or blur % 2 == 0):
        raise LabError(f"--blur deve ser 0 ou impar >= 3: {blur}")
    if method == "fixed" and threshold is None:
        raise LabError("metodo fixed exige --threshold")
    if method == "adaptive" and (block_size is None or block_size < 3 or block_size % 2 == 0):
        raise LabError(f"--block-size deve ser impar >= 3: {block_size}")
    if method == "hsv" and (hsv_lower is None or hsv_upper is None):
        raise LabError("metodo hsv exige --hsv-lower e --hsv-upper")


def segment(image_bgr, method, threshold=None, invert=False, blur=0, block_size=31, c=5.0, hsv_lower=None, hsv_upper=None):
    if method == "hsv":
        hsv_lower = parse_hsv(hsv_lower, "--hsv-lower") if hsv_lower is not None else None
        hsv_upper = parse_hsv(hsv_upper, "--hsv-upper") if hsv_upper is not None else None
    validate_params(method, threshold, block_size, c, hsv_lower, hsv_upper, blur)
    params = {"method": method, "invert": bool(invert), "blur": blur}
    if method == "hsv":
        mask = range_hsv(preprocess(image_bgr, blur), hsv_lower, hsv_upper)
        params["hsv_lower"] = list(hsv_lower)
        params["hsv_upper"] = list(hsv_upper)
        params["hue_wrap"] = hsv_lower[0] > hsv_upper[0]
    else:
        gray = preprocess(to_gray(image_bgr), blur)
        if method == "fixed":
            mask = threshold_fixed(gray, int(threshold))
            params["threshold"] = int(threshold)
        elif method == "otsu":
            mask, value = threshold_otsu(gray)
            params["otsu_threshold"] = value
        else:
            mask = threshold_adaptive(gray, int(block_size), float(c))
            params["block_size"] = int(block_size)
            params["c"] = float(c)
    if invert:
        mask = cv2.bitwise_not(mask)
    return mask.astype(np.uint8), params


def foreground_ratio(mask):
    return float(np.count_nonzero(mask == 255)) / float(mask.size)
