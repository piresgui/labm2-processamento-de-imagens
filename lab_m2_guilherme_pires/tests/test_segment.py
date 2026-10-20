import json

import cv2
import numpy as np

from pdi_lab.segment import foreground_ratio, segment


def assert_binary_mask(mask, shape):
    assert mask.shape == shape
    assert mask.ndim == 2
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 255}


def test_fixed_threshold_boundary():
    gray = np.array([[119, 120, 121]], np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask, _ = segment(image, "fixed", threshold=120)
    assert mask.tolist() == [[0, 0, 255]]


def test_fixed_square_ratio():
    image = np.full((100, 100, 3), 30, np.uint8)
    image[25:75, 25:75] = 220
    mask, _ = segment(image, "fixed", threshold=120)
    assert_binary_mask(mask, (100, 100))
    assert foreground_ratio(mask) == 0.25


def test_otsu_square_ratio():
    image = np.full((100, 100, 3), 30, np.uint8)
    image[25:75, 25:75] = 220
    mask, params = segment(image, "otsu")
    assert_binary_mask(mask, (100, 100))
    assert foreground_ratio(mask) == 0.25
    assert 30 <= params["otsu_threshold"] < 220


def test_invert_dark_object():
    image = np.full((100, 100, 3), 220, np.uint8)
    image[25:75, 25:75] = 30
    mask, _ = segment(image, "otsu", invert=True)
    assert foreground_ratio(mask) == 0.25
    assert mask[50, 50] == 255 and mask[0, 0] == 0


def test_cli_segment_ok(cli, square_image, tmp_path):
    out = tmp_path / "out" / "mask.png"
    result = cli("--input", square_image, "--output", out, "--operation", "segment", "--method", "fixed", "--threshold", 120)
    assert result.returncode == 0, result.stderr
    assert "foreground_ratio=0.250000" in result.stdout
    mask = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert_binary_mask(mask, (100, 100))
    stats = json.loads((tmp_path / "out" / "mask_stats.json").read_text())
    assert stats["foreground_ratio"] == 0.25


def test_cli_missing_file(cli, tmp_path):
    result = cli("--input", tmp_path / "nao_existe.png", "--output", tmp_path / "m.png", "--operation", "segment", "--method", "otsu")
    assert result.returncode != 0
    assert "inexistente" in result.stderr


def test_cli_invalid_format(cli, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("isto nao e imagem")
    result = cli("--input", bad, "--output", tmp_path / "m.png", "--operation", "segment", "--method", "otsu")
    assert result.returncode != 0
    assert "falha ao abrir" in result.stderr


def test_cli_unknown_method(cli, square_image, tmp_path):
    result = cli("--input", square_image, "--output", tmp_path / "m.png", "--operation", "segment", "--method", "magic")
    assert result.returncode != 0
    assert "metodo desconhecido" in result.stderr


def test_cli_threshold_out_of_range(cli, square_image, tmp_path):
    result = cli("--input", square_image, "--output", tmp_path / "m.png", "--operation", "segment", "--method", "fixed", "--threshold", 300)
    assert result.returncode != 0


def test_cli_fixed_without_threshold(cli, square_image, tmp_path):
    result = cli("--input", square_image, "--output", tmp_path / "m.png", "--operation", "segment", "--method", "fixed")
    assert result.returncode != 0


def test_cli_unknown_operation(cli, square_image, tmp_path):
    result = cli("--input", square_image, "--output", tmp_path / "m.png", "--operation", "blur")
    assert result.returncode != 0


def test_cli_jpeg_output_rejected(cli, square_image, tmp_path):
    result = cli("--input", square_image, "--output", tmp_path / "m.jpg", "--operation", "segment", "--method", "otsu")
    assert result.returncode != 0
