import cv2
import numpy as np

from pdi_lab.errors import LabError

OPERATIONS = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE, "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}
SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}
CSV_FIELDS = ("label", "area", "x", "y", "width", "height", "cx", "cy", "kept")


def parse_connectivity(value):
    if str(value) not in ("4", "8"):
        raise LabError(f"--connectivity deve ser 4 ou 8: {value}")
    return int(value)


def parse_min_area(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise LabError(f"--min-area deve ser inteiro nao negativo: {value}")
    if number < 0:
        raise LabError(f"--min-area deve ser inteiro nao negativo: {value}")
    return number


def parse_morph(spec):
    if spec is None or str(spec).strip() == "":
        return []
    steps = []
    for raw in str(spec).split(","):
        parts = raw.strip().split(":")
        if len(parts) != 4:
            raise LabError(f"passo morfologico deve ser op:forma:tamanho:iteracoes: {raw}")
        op, shape, size, iterations = parts
        if op not in OPERATIONS:
            raise LabError(f"operacao morfologica desconhecida: {op} (use {', '.join(OPERATIONS)})")
        if shape not in SHAPES:
            raise LabError(f"forma de elemento estruturante desconhecida: {shape} (use {', '.join(SHAPES)})")
        try:
            size, iterations = int(size), int(iterations)
        except ValueError:
            raise LabError(f"tamanho e iteracoes devem ser inteiros: {raw}")
        if size < 1 or size % 2 == 0:
            raise LabError(f"tamanho do elemento estruturante deve ser impar >= 1: {size}")
        if iterations < 1:
            raise LabError(f"iteracoes deve ser >= 1: {iterations}")
        steps.append({"op": op, "shape": shape, "size": size, "iterations": iterations})
    return steps


def sequence_warnings(steps):
    warnings = []
    if len(steps) < 2:
        warnings.append("sequencia com menos de duas operacoes")
    if not any(s["op"] in ("open", "close") for s in steps):
        warnings.append("sequencia sem abertura ou fechamento")
    for a, b in zip(steps, steps[1:]):
        if a == b:
            warnings.append(f"operacao repetida com os mesmos parametros: {a['op']}")
    return warnings


def apply_step(mask, step):
    kernel = cv2.getStructuringElement(SHAPES[step["shape"]], (step["size"], step["size"]))
    return cv2.morphologyEx(mask, OPERATIONS[step["op"]], kernel, iterations=step["iterations"])


def run_sequence(mask, steps):
    stages = [("initial", mask)]
    current = mask
    for index, step in enumerate(steps, start=1):
        current = apply_step(current, step)
        stages.append((f"{index}_{step['op']}_{step['shape']}{step['size']}x{step['iterations']}", current))
    return stages


def label_components(mask, connectivity, min_area=0):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    components = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        components.append({
            "label": label,
            "area": area,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "cx": round(float(centroids[label][0]), 2),
            "cy": round(float(centroids[label][1]), 2),
            "kept": area >= min_area,
        })
    return components, labels


def count_summary(mask, connectivity, min_area):
    components, _ = label_components(mask, connectivity, min_area)
    kept = [c for c in components if c["kept"]]
    return {
        "foreground_pixels": int((mask == 255).sum()),
        "components_conn4": len(label_components(mask, 4)[0]),
        "components_conn8": len(label_components(mask, 8)[0]),
        "raw_count": len(components),
        "filtered_count": len(kept),
        "raw_area_total": sum(c["area"] for c in components),
        "kept_area_total": sum(c["area"] for c in kept),
    }


def label_color(label):
    rng = np.random.default_rng(label * 7919)
    return tuple(int(v) for v in rng.integers(60, 256, 3))


def render_labels(mask, components, labels):
    canvas = np.zeros((*mask.shape, 3), np.uint8)
    thickness = max(1, mask.shape[1] // 500)
    for comp in components:
        color = label_color(comp["label"]) if comp["kept"] else (90, 90, 90)
        canvas[labels == comp["label"]] = color
    for comp in components:
        box_color = (0, 255, 0) if comp["kept"] else (0, 0, 255)
        cv2.rectangle(canvas, (comp["x"], comp["y"]), (comp["x"] + comp["width"] - 1, comp["y"] + comp["height"] - 1), box_color, thickness)
        if comp["kept"]:
            cv2.putText(canvas, str(comp["label"]), (comp["x"], max(10, comp["y"] - 3)), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, mask.shape[1] / 1500), (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def analyze(mask, connectivity, min_area, steps):
    stages = run_sequence(mask, steps)
    final_mask = stages[-1][1]
    initial_components, initial_labels = label_components(mask, connectivity, min_area)
    final_components, final_labels = label_components(final_mask, connectivity, min_area)
    return {
        "stages": stages,
        "final_mask": final_mask,
        "initial_components": initial_components,
        "initial_labels": initial_labels,
        "final_components": final_components,
        "final_labels": final_labels,
        "initial_summary": count_summary(mask, connectivity, min_area),
        "final_summary": count_summary(final_mask, connectivity, min_area),
    }
