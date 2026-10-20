from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "images" / "synthetic"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    square = np.full((100, 100, 3), 30, np.uint8)
    square[25:75, 25:75] = 220
    cv2.imwrite(str(OUT / "square_light.png"), square)
    gt = np.zeros((100, 100), np.uint8)
    gt[25:75, 25:75] = 255
    cv2.imwrite(str(OUT / "square_light_gt.png"), gt)

    scene = np.full((160, 240, 3), 200, np.uint8)
    ramp = np.tile(np.linspace(-40, 40, 240).astype(np.int16), (160, 1))
    scene = np.clip(scene.astype(np.int16) + ramp[:, :, None], 0, 255).astype(np.uint8)
    scene_gt = np.zeros((160, 240), np.uint8)
    for cx, cy, r in [(50, 60, 22), (120, 90, 28), (195, 55, 18), (190, 120, 15)]:
        cv2.circle(scene, (cx, cy), r, (40, 90, 200), -1)
        cv2.circle(scene_gt, (cx, cy), r, 255, -1)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 12, scene.shape)
    scene = np.clip(scene.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / "coins_scene.png"), scene)
    cv2.imwrite(str(OUT / "coins_scene_gt.png"), scene_gt)
    print(f"imagens sinteticas salvas em {OUT.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
