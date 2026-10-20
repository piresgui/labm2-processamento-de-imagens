import argparse
import csv
import json
import sys
from pathlib import Path

from pdi_lab import analyze as analysis
from pdi_lab import imageio, visualize
from pdi_lab.errors import LabError
from pdi_lab.segment import foreground_ratio, segment

SEGMENT_KEYS = ("method", "threshold", "invert", "blur", "block_size", "c", "hsv_lower", "hsv_upper")


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise LabError(message)


def build_parser():
    parser = ArgumentParser(prog="pdi_lab", description="Laboratorio M2 - segmentacao, componentes conexos e morfologia")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--method", default="fixed")
    parser.add_argument("--threshold", type=int)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--blur", type=int, default=0)
    parser.add_argument("--block-size", dest="block_size", type=int, default=31)
    parser.add_argument("--c", type=float, default=5.0)
    parser.add_argument("--hsv-lower", dest="hsv_lower")
    parser.add_argument("--hsv-upper", dest="hsv_upper")
    parser.add_argument("--reference")
    parser.add_argument("--config")
    parser.add_argument("--connectivity", default="8")
    parser.add_argument("--min-area", dest="min_area", default="0")
    parser.add_argument("--morph", default="")
    parser.add_argument("--original")
    return parser


def segment_kwargs(source):
    return {k: source[k] for k in SEGMENT_KEYS if k in source and source[k] is not None}


def mask_stats(input_path, output_path, mask, params):
    return {
        "input": Path(input_path).as_posix(),
        "output": Path(output_path).as_posix(),
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "foreground_pixels": int((mask == 255).sum()),
        "total_pixels": int(mask.size),
        "foreground_ratio": foreground_ratio(mask),
        "params": params,
    }


def load_reference(path, shape):
    if path is None:
        return None
    reference = imageio.load_mask(path)
    if reference.shape != shape:
        raise LabError(f"mascara de referencia com tamanho {reference.shape}, esperado {shape}")
    return reference


def run_segment(args):
    image = imageio.load_color(args.input)
    mask, params = segment(image, **segment_kwargs(vars(args)))
    reference = load_reference(args.reference, mask.shape)
    imageio.save_mask(args.output, mask)
    stats = mask_stats(args.input, args.output, mask, params)
    out = Path(args.output)
    if reference is not None:
        stats["reference"] = Path(args.reference).as_posix()
        stats.update(visualize.compare_reference(mask, reference))
        imageio.save_image(out.with_name(out.stem + "_errors.png"), visualize.error_map(image, mask, reference))
    out.with_name(out.stem + "_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"width={stats['width']}")
    print(f"height={stats['height']}")
    print(f"foreground_pixels={stats['foreground_pixels']}")
    print(f"foreground_ratio={stats['foreground_ratio']:.6f}")
    if reference is not None:
        print(f"fp={stats['fp']} fn={stats['fn']} iou={stats['iou']:.4f}")
    print(f"params={json.dumps(params)}")
    return 0


def load_config(path, image_name):
    if path is None:
        raise LabError("operacao exige --config (arquivo JSON de configuracoes)")
    p = Path(path)
    if not p.is_file():
        raise LabError(f"arquivo de configuracao inexistente: {path}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError(f"configuracao JSON invalida: {exc}")
    merged = dict(data.get("default", {}))
    merged.update(data.get("images", {}).get(image_name, {}))
    return merged


def run_compare(args):
    image = imageio.load_color(args.input)
    config = load_config(args.config, Path(args.input).name)
    configs = config.get("configs")
    if not configs:
        raise LabError("configuracao sem lista 'configs'")
    reference_path = args.reference or config.get("reference")
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    figure_rows = []
    reference = None
    for item in configs:
        name = item.get("name")
        if not name:
            raise LabError("cada configuracao precisa de 'name'")
        mask, params = segment(image, **segment_kwargs(item))
        if reference is None and reference_path:
            reference = load_reference(reference_path, mask.shape)
        mask_path = prefix.with_name(f"{prefix.name}_{name}_mask.png")
        overlay_img = visualize.overlay(image, mask)
        imageio.save_mask(mask_path, mask)
        imageio.save_image(prefix.with_name(f"{prefix.name}_{name}_overlay.png"), overlay_img)
        row = {
            "image": Path(args.input).name,
            "config": name,
            "params": json.dumps(params),
            "foreground_pixels": int((mask == 255).sum()),
            "total_pixels": int(mask.size),
            "foreground_ratio": round(foreground_ratio(mask), 6),
        }
        figure_row = [(image, "original"), (mask, f"mascara {name}"), (overlay_img, f"sobreposicao {name}")]
        if reference is not None:
            metrics = visualize.compare_reference(mask, reference)
            row.update({"fp": metrics["fp"], "fn": metrics["fn"], "iou": round(metrics["iou"], 4)})
            errors = visualize.error_map(image, mask, reference)
            imageio.save_image(prefix.with_name(f"{prefix.name}_{name}_errors.png"), errors)
            figure_row.append((errors, f"erros {name} (FP verm, FN azul)"))
        rows.append(row)
        figure_rows.append(figure_row)
        print(f"{name}: foreground_ratio={row['foreground_ratio']:.6f}" + (f" fp={row['fp']} fn={row['fn']} iou={row['iou']:.4f}" if reference is not None else ""))
    csv_path = prefix.with_name(f"{prefix.name}_compare.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    imageio.save_image(prefix.with_name(f"{prefix.name}_compare_panel.png"), visualize.panel(figure_rows))
    print(f"tabela={csv_path.as_posix()}")
    return 0


def write_components_csv(path, components):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis.CSV_FIELDS))
        writer.writeheader()
        for comp in components:
            writer.writerow({**comp, "kept": str(comp["kept"]).lower()})


def run_analyze(args):
    connectivity = analysis.parse_connectivity(args.connectivity)
    min_area = analysis.parse_min_area(args.min_area)
    steps = analysis.parse_morph(args.morph)
    mask = imageio.load_mask(args.input)
    original = imageio.load_color(args.original) if args.original else None
    if original is not None and original.shape[:2] != mask.shape:
        raise LabError(f"--original com tamanho {original.shape[:2]} diferente da mascara {mask.shape}")
    result = analysis.analyze(mask, connectivity, min_area, steps)
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    name = lambda suffix: prefix.with_name(f"{prefix.name}_{suffix}")
    stage_files = []
    for index, (label, image) in enumerate(result["stages"]):
        path = name(f"stage{index}_{label.split('_', 1)[-1] if index else 'initial'}.png")
        imageio.save_mask(path, image)
        stage_files.append(path.as_posix())
    imageio.save_mask(name("final_mask.png"), result["final_mask"])
    write_components_csv(name("initial.csv"), result["initial_components"])
    write_components_csv(name("final.csv"), result["final_components"])
    initial_labels = analysis.render_labels(mask, result["initial_components"], result["initial_labels"])
    final_labels = analysis.render_labels(result["final_mask"], result["final_components"], result["final_labels"])
    imageio.save_image(name("initial_labels.png"), initial_labels)
    imageio.save_image(name("labels.png"), final_labels)
    stage_row = ([(original, "original")] if original is not None else []) + [(img, f"estagio {i}: {lbl}") for i, (lbl, img) in enumerate(result["stages"])]
    imageio.save_image(name("stages_panel.png"), visualize.panel([stage_row, [(initial_labels, "rotulos iniciais"), (final_labels, "rotulos finais")]]))
    initial, final = result["initial_summary"], result["final_summary"]
    with name("before_after.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "initial", "final"])
        for key in initial:
            writer.writerow([key, initial[key], final[key]])
    warnings = analysis.sequence_warnings(steps)
    summary = {
        "input": Path(args.input).as_posix(),
        "original": Path(args.original).as_posix() if args.original else None,
        "connectivity": connectivity,
        "min_area": min_area,
        "filter_rule": f"componente mantido se area >= {min_area} pixels",
        "morphology": steps,
        "warnings": warnings,
        "initial": initial,
        "final": final,
        "stage_files": stage_files,
    }
    name("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for warning in warnings:
        print(f"aviso: {warning}", file=sys.stderr)
    print(f"connectivity={connectivity} min_area={min_area}")
    print(f"initial: conn4={initial['components_conn4']} conn8={initial['components_conn8']} raw={initial['raw_count']} filtered={initial['filtered_count']} area={initial['foreground_pixels']}")
    print(f"final: conn4={final['components_conn4']} conn8={final['components_conn8']} raw={final['raw_count']} filtered={final['filtered_count']} area={final['foreground_pixels']}")
    print(f"summary={name('summary.json').as_posix()}")
    return 0


HANDLERS = {"segment": run_segment, "compare": run_compare, "analyze": run_analyze}


def main(argv=None):
    try:
        args = build_parser().parse_args(argv)
        if args.operation not in HANDLERS:
            raise LabError(f"operacao desconhecida: {args.operation} (use {', '.join(HANDLERS)})")
        return HANDLERS[args.operation](args)
    except LabError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
