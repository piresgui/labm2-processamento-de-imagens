from pathlib import Path

import cv2
import numpy as np

from pdi_lab.errors import LabError

LOSSLESS_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff", ".pgm", ".ppm"}


def _decode(path, flags, kind):
    p = Path(path)
    if not p.is_file():
        raise LabError(f"arquivo de {kind} inexistente: {path}")
    data = np.fromfile(str(p), dtype=np.uint8)
    image = cv2.imdecode(data, flags) if data.size else None
    if image is None:
        raise LabError(f"falha ao abrir {kind} (formato invalido ou arquivo corrompido): {path}")
    return image


def load_color(path):
    image = _decode(path, cv2.IMREAD_COLOR, "entrada")
    return image


def load_mask(path):
    image = _decode(path, cv2.IMREAD_UNCHANGED, "mascara")
    if image.ndim != 2:
        raise LabError(f"mascara deve ter 1 canal, encontrado shape {image.shape}: {path}")
    if image.dtype != np.uint8:
        raise LabError(f"mascara deve ser uint8, encontrado {image.dtype}: {path}")
    if not set(np.unique(image).tolist()) <= {0, 255}:
        raise LabError(f"mascara deve conter apenas 0 e 255: {path}")
    return image


def save_image(path, image):
    p = Path(path)
    if p.suffix.lower() not in LOSSLESS_EXTENSIONS:
        raise LabError(f"formato de saida deve ser sem perdas (png, bmp, tif): {path}")
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(p.suffix.lower(), image)
    if not ok:
        raise LabError(f"falha ao salvar imagem: {path}")
    buffer.tofile(str(p))


def save_mask(path, mask):
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise LabError("mascara interna invalida")
    if not set(np.unique(mask).tolist()) <= {0, 255}:
        raise LabError("mascara interna nao binaria")
    save_image(path, mask)
