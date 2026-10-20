import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "pdi_lab", *map(str, args)], capture_output=True, text=True, env=env, cwd=ROOT)


@pytest.fixture
def cli():
    return run_cli


@pytest.fixture
def square_image(tmp_path):
    image = np.full((100, 100, 3), 30, np.uint8)
    image[25:75, 25:75] = 220
    path = tmp_path / "square.png"
    cv2.imwrite(str(path), image)
    return path
