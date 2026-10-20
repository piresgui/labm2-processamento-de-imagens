import csv
import json

import cv2
import numpy as np
import pytest

from pdi_lab.analyze import analyze, label_components, parse_morph, run_sequence
from pdi_lab.errors import LabError


def save(path, mask):
    cv2.imwrite(str(path), mask)
    return path


def test_diagonal_pixels_conn4_vs_conn8():
    mask = np.zeros((5, 5), np.uint8)
    mask[1, 1] = 255
    mask[2, 2] = 255
    assert len(label_components(mask, 4)[0]) == 2
    assert len(label_components(mask, 8)[0]) == 1


def test_opening_removes_isolated_noise():
    mask = np.zeros((60, 60), np.uint8)
    mask[15:45, 15:45] = 255
    mask[3, 3] = 255
    mask[55, 50] = 255
    stages = run_sequence(mask, parse_morph("open:rect:3:1"))
    final = stages[-1][1]
    assert len(label_components(mask, 8)[0]) == 3
    assert len(label_components(final, 8)[0]) == 1
    assert (final[15:45, 15:45] == 255).all()


def test_closing_fills_small_hole():
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = 255
    mask[19:21, 19:21] = 0
    final = run_sequence(mask, parse_morph("close:rect:3:1"))[-1][1]
    assert (final[10:30, 10:30] == 255).all()


def test_empty_mask_zero_components():
    result = analyze(np.zeros((10, 10), np.uint8), 8, 0, parse_morph("open:rect:3:1,close:rect:3:1"))
    assert result["initial_summary"]["raw_count"] == 0
    assert result["final_summary"]["raw_count"] == 0


def test_object_touching_border_survives_opening():
    mask = np.zeros((30, 30), np.uint8)
    mask[0:12, 0:12] = 255
    result = analyze(mask, 8, 0, parse_morph("open:rect:3:1"))
    comp = result["final_components"][0]
    assert (comp["x"], comp["y"], comp["area"]) == (0, 0, 144)


def test_min_area_filter_marks_kept():
    mask = np.zeros((30, 30), np.uint8)
    mask[2:12, 2:12] = 255
    mask[20:22, 20:22] = 255
    components, _ = label_components(mask, 8, min_area=10)
    assert sorted((c["area"], c["kept"]) for c in components) == [(4, False), (100, True)]


@pytest.mark.parametrize("spec", ["open:rect:4:1", "blur:rect:3:1", "open:star:3:1", "open:rect:3", "open:rect:3:0"])
def test_invalid_morph(spec):
    with pytest.raises(LabError):
        parse_morph(spec)


def test_cli_analyze_outputs(cli, tmp_path):
    mask = np.zeros((50, 50), np.uint8)
    mask[5:20, 5:20] = 255
    mask[30:45, 30:45] = 255
    mask[2, 40] = 255
    path = save(tmp_path / "mask.png", mask)
    prefix = tmp_path / "res" / "cena"
    result = cli("--input", path, "--output", prefix, "--operation", "analyze", "--connectivity", 8, "--min-area", 20, "--morph", "open:rect:3:1,close:rect:3:1")
    assert result.returncode == 0, result.stderr
    with (tmp_path / "res" / "cena_initial.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == ["label", "area", "x", "y", "width", "height", "cx", "cy", "kept"]
    assert sum(int(r["area"]) for r in rows) == int((mask == 255).sum())
    assert sorted(r["kept"] for r in rows) == ["false", "true", "true"]
    summary = json.loads((tmp_path / "res" / "cena_summary.json").read_text())
    assert summary["connectivity"] == 8
    assert summary["initial"]["raw_count"] == 3 and summary["initial"]["filtered_count"] == 2
    assert summary["final"]["raw_count"] == 2
    for suffix in ("final.csv", "labels.png", "stage0_initial.png", "stage1_open_rect3x1.png", "stage2_close_rect3x1.png", "stages_panel.png", "before_after.csv"):
        assert (tmp_path / "res" / f"cena_{suffix}").is_file(), suffix


def test_cli_analyze_invalid_connectivity(cli, tmp_path):
    path = save(tmp_path / "mask.png", np.zeros((5, 5), np.uint8))
    result = cli("--input", path, "--output", tmp_path / "p", "--operation", "analyze", "--connectivity", 6)
    assert result.returncode != 0
    assert "connectivity" in result.stderr


def test_cli_analyze_negative_min_area(cli, tmp_path):
    path = save(tmp_path / "mask.png", np.zeros((5, 5), np.uint8))
    result = cli("--input", path, "--output", tmp_path / "p", "--operation", "analyze", "--min-area", -1)
    assert result.returncode != 0


def test_cli_analyze_rejects_non_binary(cli, tmp_path):
    path = save(tmp_path / "gray.png", np.full((5, 5), 128, np.uint8))
    result = cli("--input", path, "--output", tmp_path / "p", "--operation", "analyze")
    assert result.returncode != 0
    assert "apenas 0 e 255" in result.stderr
