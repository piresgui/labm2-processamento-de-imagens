import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def cli_args(config):
    args = ["--method", config["method"]]
    for key in ("threshold", "blur", "block_size", "c", "hsv_lower", "hsv_upper"):
        if key in config:
            args += [f"--{key.replace('_', '-')}", str(config[key])]
    if config.get("invert"):
        args.append("--invert")
    return args


def run(args, log):
    command = [sys.executable, "-m", "pdi_lab", *args]
    log.append("python -m pdi_lab " + " ".join(shlex.quote(a) for a in args))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    print(result.stderr, end="", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Processa todas as imagens de uma pasta (M2.1 + M2.2)")
    parser.add_argument("--input-dir", default="images/input")
    parser.add_argument("--config", default="configs/pipeline.json")
    parser.add_argument("--results", default="results")
    parser.add_argument("--masks", default="images/output")
    opts = parser.parse_args()
    data = json.loads((ROOT / opts.config).read_text(encoding="utf-8"))
    images = sorted(p for p in (ROOT / opts.input_dir).iterdir() if p.suffix.lower() in EXTENSIONS and not p.stem.endswith("_gt"))
    if not images:
        print(f"nenhuma imagem encontrada em {opts.input_dir}", file=sys.stderr)
        return 1
    log = []
    compare_rows = []
    analyze_rows = []
    for image in images:
        cfg = {**data.get("default", {}), **data.get("images", {}).get(image.name, {})}
        stem = image.stem
        out_dir = ROOT / opts.results / stem
        print(f"== {image.name}")
        compare_args = ["--input", rel(image), "--output", f"{opts.results}/{stem}/{stem}", "--operation", "compare", "--config", opts.config]
        if cfg.get("reference"):
            compare_args += ["--reference", cfg["reference"]]
        run(compare_args, log)
        with (out_dir / f"{stem}_compare.csv").open(encoding="utf-8") as handle:
            compare_rows += list(csv.DictReader(handle))
        chosen = next(c for c in cfg["configs"] if c["name"] == cfg["chosen"])
        mask_path = f"{opts.masks}/{stem}_mask.png"
        run(["--input", rel(image), "--output", mask_path, "--operation", "segment", *cli_args(chosen)], log)
        a = cfg["analyze"]
        run(["--input", mask_path, "--output", f"{opts.results}/{stem}/{stem}", "--operation", "analyze", "--connectivity", str(a["connectivity"]), "--min-area", str(a["min_area"]), "--morph", a["morph"], "--original", rel(image)], log)
        summary = json.loads((out_dir / f"{stem}_summary.json").read_text(encoding="utf-8"))
        for stage in ("initial", "final"):
            analyze_rows.append({"image": image.name, "stage": stage, "chosen_config": cfg["chosen"], "morph": a["morph"], "connectivity": a["connectivity"], "min_area": a["min_area"], **summary[stage]})
    results = ROOT / opts.results
    for name, rows in (("summary_compare.csv", compare_rows), ("summary_analyze.csv", analyze_rows)):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (results / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (results / "commands.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"tabelas: {opts.results}/summary_compare.csv, {opts.results}/summary_analyze.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
