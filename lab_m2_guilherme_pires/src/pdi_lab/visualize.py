import cv2
import numpy as np

from pdi_lab.errors import LabError

TILE_WIDTH = 360
GREEN = (0, 200, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)


def overlay(image_bgr, mask, alpha=0.45):
    tinted = image_bgr.copy()
    tinted[mask == 255] = GREEN
    result = cv2.addWeighted(tinted, alpha, image_bgr, 1 - alpha, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(result, contours, -1, YELLOW, max(1, image_bgr.shape[1] // 400))
    return result


def compare_reference(mask, reference):
    if mask.shape != reference.shape:
        raise LabError(f"mascara de referencia com tamanho diferente: {reference.shape} vs {mask.shape}")
    pred = mask == 255
    ref = reference == 255
    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    union = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "iou": tp / union if union else 1.0}


def error_map(image_bgr, mask, reference):
    gray = cv2.cvtColor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    result = (gray * 0.5).astype(np.uint8)
    pred = mask == 255
    ref = reference == 255
    result[pred & ref] = GREEN
    result[pred & ~ref] = RED
    result[~pred & ref] = BLUE
    return result


def tile(image, label):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = TILE_WIDTH / image.shape[1]
    resized = cv2.resize(image, (TILE_WIDTH, max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    header = np.full((28, TILE_WIDTH, 3), 255, np.uint8)
    cv2.putText(header, label[:48], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([header, resized])


def panel(rows):
    width = max(len(r) for r in rows)
    built = []
    for row in rows:
        tiles = [tile(img, label) for img, label in row]
        while len(tiles) < width:
            tiles.append(np.full_like(tiles[0], 255))
        built.append(np.hstack(tiles))
    return np.vstack(built)
