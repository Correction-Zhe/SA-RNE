"""
Generate synthetic residual stripe noise for grayscale infrared images.

The simulator creates sparse vertical stripe segments with small high-frequency
Gaussian perturbations, then suppresses injected noise around strong edges.
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


INPUT_PATH = r"/IR_Denoising/DAG_CNN_V2/datasets/clean"
OUTPUT_PATH = r"/IR_Denoising/DAG_CNN_V2/datasets/synthetic_level1"
DIFF_DIR = r"/IR_Denoising/DAG_CNN_V2/datasets/noisy_gtvcnn/diff"
PREVIEW_DIR = r"/IR_Denoising/DAG_CNN_V2/datasets/noisy_gtvcnn/preview"

COL_FRAC_MIN = 0.70
COL_FRAC_MAX = 0.99
SEG_H_MIN = 30
SEG_H_MAX = 50
STRIPE_AMP = 0.05

GAUSSIAN_SIGMA_MIN = 0.010
GAUSSIAN_SIGMA_MAX = 0.020

EDGE_SUPPRESS_THR = 0.15
EDGE_SUPPRESS_W = 0.15

SEED = 42
SAVE_DIFF = False
SAVE_PREVIEW = False

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _edge_weight_map(image: np.ndarray) -> np.ndarray:
    """Return a [0, 1] noise-injection weight map with lower weights near edges."""
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag = mag / (mag.max() + 1e-6)

    weight = np.where(
        mag > EDGE_SUPPRESS_THR,
        EDGE_SUPPRESS_W,
        1.0 - (1.0 - EDGE_SUPPRESS_W) * (mag / EDGE_SUPPRESS_THR),
    ).astype(np.float32)
    return weight


def _build_clahe_residual_noise(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a synthetic residual stripe noise map."""
    noise = np.zeros((height, width), dtype=np.float32)

    col_frac = rng.uniform(COL_FRAC_MIN, COL_FRAC_MAX)
    n_cols = max(1, int(width * col_frac))
    cols = rng.choice(width, size=n_cols, replace=False)

    for col in cols:
        row = 0
        while row < height:
            seg_h = int(rng.integers(SEG_H_MIN, SEG_H_MAX + 1))
            seg_h = min(seg_h, height - row)

            if rng.random() < 0.6:
                dc_offset = rng.uniform(-STRIPE_AMP, STRIPE_AMP)
                sigma = rng.uniform(GAUSSIAN_SIGMA_MIN, GAUSSIAN_SIGMA_MAX)
                hf_noise = rng.normal(0.0, sigma, size=seg_h).astype(np.float32)
                noise[row: row + seg_h, col] = dc_offset + hf_noise

            gap = int(rng.integers(0, SEG_H_MIN))
            row += seg_h + gap

    return noise


def add_clahe_residual_noise(
    image: np.ndarray,
    seed: Optional[int] = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add synthetic residual stripe noise to a grayscale image.

    Parameters
    ----------
    image:
        Float32 array with shape (H, W) and values in [0, 1].

    Returns
    -------
    noisy:
        Float32 noisy image clipped to [0, 1].
    diff:
        Signed residual map, computed as noisy - image.
    """
    if image.ndim != 2:
        raise ValueError("Expected a 2-D grayscale array.")
    if image.dtype != np.float32:
        raise TypeError("Expected float32 input.")

    rng = np.random.default_rng(seed)
    height, width = image.shape
    raw_noise = _build_clahe_residual_noise(height, width, rng)
    noise = raw_noise * _edge_weight_map(image)

    noisy = np.clip(image + noise, 0.0, 1.0).astype(np.float32)
    diff = noisy - image
    return noisy, diff


def load_gray_float(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img.astype(np.float32) / 255.0


def save_gray_float(array: np.ndarray, path: str) -> None:
    cv2.imwrite(path, (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8))


def save_diff_map(diff: np.ndarray, path: str) -> None:
    abs_max = max(float(np.abs(diff).max()), 1e-6)
    norm = (diff / abs_max) * 0.5 + 0.5
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(path, color)


def save_preview(
    original: np.ndarray,
    noisy: np.ndarray,
    diff: np.ndarray,
    save_path: str,
    fname: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping preview.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"CLAHE Residual Noise Simulation - {fname}", fontsize=12)

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Noisy")
    axes[1].axis("off")

    amp = np.clip(diff * 15 + 0.5, 0, 1)
    axes[2].imshow(amp, cmap="RdBu_r", vmin=0, vmax=1)
    axes[2].set_title("Diff x15")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def collect_paths(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if os.path.splitext(name)[1].lower() in SUPPORTED_EXT
        )
    raise FileNotFoundError(f"Input not found: {input_path}")


def process_one(
    img_path: str,
    out_dir: str,
    diff_dir: Optional[str],
    preview_dir: Optional[str],
) -> None:
    fname = os.path.basename(img_path)
    stem = os.path.splitext(fname)[0]
    original = load_gray_float(img_path)

    noisy, diff = add_clahe_residual_noise(original)
    save_gray_float(noisy, os.path.join(out_dir, fname))

    if SAVE_DIFF and diff_dir:
        save_diff_map(diff, os.path.join(diff_dir, f"{stem}_diff.png"))

    if SAVE_PREVIEW and preview_dir:
        save_preview(
            original,
            noisy,
            diff,
            os.path.join(preview_dir, f"{stem}_preview.png"),
            fname,
        )

    mse = max(float(np.mean((noisy - original) ** 2)), 1e-10)
    psnr = 10 * np.log10(1.0 / mse)
    print(f"  {fname:<40} PSNR={psnr:6.2f} dB |diff|_max={float(np.abs(diff).max()):.4f}")


def main() -> None:
    paths = collect_paths(INPUT_PATH)
    if not paths:
        raise FileNotFoundError(f"No images found in: {INPUT_PATH}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    diff_dir = DIFF_DIR if SAVE_DIFF else None
    preview_dir = PREVIEW_DIR if SAVE_PREVIEW else None
    if diff_dir:
        os.makedirs(diff_dir, exist_ok=True)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

    print("=" * 65)
    print(f"Input            : {INPUT_PATH}")
    print(f"Output           : {OUTPUT_PATH}")
    print(f"Images found     : {len(paths)}")
    print(f"Col frac range   : {COL_FRAC_MIN}-{COL_FRAC_MAX}")
    print(f"Seg height range : {SEG_H_MIN}-{SEG_H_MAX} px")
    print(f"Stripe amp       : +/-{STRIPE_AMP}")
    print(f"Gaussian sigma   : {GAUSSIAN_SIGMA_MIN}-{GAUSSIAN_SIGMA_MAX}")
    print(f"Edge suppress    : thr={EDGE_SUPPRESS_THR} weight={EDGE_SUPPRESS_W}")
    print(f"Seed             : {SEED}")
    print("=" * 65)

    for index, path in enumerate(paths, start=1):
        print(f"[{index:03d}/{len(paths)}]", end=" ")
        process_one(path, OUTPUT_PATH, diff_dir, preview_dir)

    print("=" * 65)
    print("Done.")
    print(f"  Noisy images -> {OUTPUT_PATH}")
    if diff_dir:
        print(f"  Diff maps    -> {diff_dir}")
    if preview_dir:
        print(f"  Previews     -> {preview_dir}")


if __name__ == "__main__":
    main()
