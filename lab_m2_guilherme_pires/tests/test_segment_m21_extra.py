import csv
import json

import cv2
import numpy as np
import pytest

from pdi_lab.errors import LabError
from pdi_lab.segment import foreground_ratio, segment
from pdi_lab.visualize import compare_reference, error_map


def test_adaptive_square_on_gradient():
    ramp = np.tile(np.linspace(20, 120, 120).astype(np.uint8), (120, 1))
    gray = ramp.copy()
    gray[40:80, 40:80] = np.clip(ramp[40:80, 40:80].astype(int) + 100, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask, params = segment(image, "adaptive", block_size=81, c=-20)
    assert params["block_size"] == 81
    assert mask[60, 60] == 255
    assert mask[5, 5] == 0 and mask[5, 115] == 0
    assert set(np.unique(mask).tolist()) <= {0, 255}


def test_hsv_range_and_ratio():
    image = np.zeros((50, 50, 3), np.uint8)
    image[:, :25] = (0, 0, 255)
    image[:, 25:] = (255, 0, 0)
    mask, _ = segment(image, "hsv", hsv_lower="0,100,100", hsv_upper="10,255,255")
    assert foreground_ratio(mask) == 0.5
    assert mask[0, 0] == 255 and mask[0, 49] == 0


def test_hsv_hue_wrap():
    image = np.zeros((10, 30, 3), np.uint8)
    hsv = np.zeros((10, 30, 3), np.uint8)
    hsv[:, :10] = (175, 255, 255)
    hsv[:, 10:20] = (3, 255, 255)
    hsv[:, 20:] = (60, 255, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    mask, params = segment(image, "hsv", hsv_lower="170,100,100", hsv_upper="10,255,255")
    assert params["hue_wrap"] is True
    assert mask[:, :20].min() == 255
    assert mask[:, 20:].max() == 0


@pytest.mark.parametrize("lower", ["200,0,0", "1,2", "a,b,c"])
def test_hsv_invalid(lower):
    image = np.zeros((5, 5, 3), np.uint8)
    with pytest.raises(LabError):
        segment(image, "hsv", hsv_lower=lower, hsv_upper="10,255,255")


def test_adaptive_even_block_invalid():
    with pytest.raises(LabError):
        segment(np.zeros((5, 5, 3), np.uint8), "adaptive", block_size=10)


def test_reference_metrics_and_error_map():
    mask = np.zeros((4, 4), np.uint8)
    mask[0, :3] = 255
    ref = np.zeros((4, 4), np.uint8)
    ref[0, 1:4] = 255
    m = compare_reference(mask, ref)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert m["iou"] == pytest.approx(0.5)
    emap = error_map(np.zeros((4, 4, 3), np.uint8), mask, ref)
    assert tuple(emap[0, 0]) == (0, 0, 255)
    assert tuple(emap[0, 3]) == (255, 0, 0)


def test_cli_compare(cli, square_image, tmp_path):
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps({"default": {"configs": [
        {"name": "fixed", "method": "fixed", "threshold": 120},
        {"name": "otsu_inv", "method": "otsu", "invert": True}]}}))
    prefix = tmp_path / "res" / "sq"
    result = cli("--input", square_image, "--output", prefix, "--operation", "compare", "--config", config)
    assert result.returncode == 0, result.stderr
    with (tmp_path / "res" / "sq_compare.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [float(r["foreground_ratio"]) for r in rows] == [0.25, 0.75]
    assert (tmp_path / "res" / "sq_compare_panel.png").is_file()


def test_cli_segment_with_reference(cli, square_image, tmp_path):
    ref = np.zeros((100, 100), np.uint8)
    ref[25:75, 25:75] = 255
    ref_path = tmp_path / "ref.png"
    cv2.imwrite(str(ref_path), ref)
    result = cli("--input", square_image, "--output", tmp_path / "m.png", "--operation", "segment", "--method", "otsu", "--reference", ref_path)
    assert result.returncode == 0, result.stderr
    assert "fp=0 fn=0 iou=1.0000" in result.stdout
    assert (tmp_path / "m_errors.png").is_file()


def test_cli_compare_missing_config(cli, square_image, tmp_path):
    result = cli("--input", square_image, "--output", tmp_path / "p", "--operation", "compare", "--config", tmp_path / "x.json")
    assert result.returncode != 0
