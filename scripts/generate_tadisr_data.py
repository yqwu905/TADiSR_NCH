"""Generate a small TADiSR-style dataset for Colab GPU smoke training.

Produces three aligned PNG folders under ``--output-dir``::

    {output_dir}/HR/000000.png   # background with random overlay text (GT)
    {output_dir}/LR/000000.png   # real-world degraded version, upsampled back
    {output_dir}/Mask/000000.png  # grayscale text-region mask

Backgrounds come from the openly licensed DIV2K train split (~800 2K
images, CC BY-NC-SA / non-commercial research). Fonts are pulled from
the OFL-licensed subset of Google Fonts via direct raw-file download
plus a fallback to system fonts installed by apt in Colab.

Real-world degradation pipeline (applied per image with random params):
    bicubic downsample (lr_size) -> gaussian blur -> JPEG encode/decode
    -> additive gaussian noise -> bicubic upsample to hr_size

Usage (typical Colab invocation)::

    python scripts/generate_tadisr_data.py \
        --output-dir /content/tadisr_dataset \
        --num-images 1000 \
        --hr-size 1024 \
        --lr-size 256 \
        --seed 42
"""
from __future__ import annotations

import argparse
import io
import os
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DIV2K_TRAIN_URL = (
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
)

# A small curated subset of OFL fonts hosted in github.com/google/fonts.
# Each entry is (raw_url, recommended_pixel_height_for_1024_image).
EXTRA_FONTS = [
    ("https://github.com/google/fonts/raw/main/ofl/roboto/"
     "Roboto%5Bwdth,wght%5D.ttf", 64),
    ("https://github.com/google/fonts/raw/main/ofl/montserrat/"
     "Montserrat%5Bwdth,wght%5D.ttf", 64),
    ("https://github.com/google/fonts/raw/main/ofl/lato/"
     "Lato%5Bwdth,wght%5D.ttf", 60),
    ("https://github.com/google/fonts/raw/main/ofl/oswald/"
     "Oswald%5Bwdth,wght%5D.ttf", 80),
    ("https://github.com/google/fonts/raw/main/ofl/raleway/"
     "Raleway%5Bwdth,wght%5D.ttf", 56),
    ("https://github.com/google/fonts/raw/main/ofl/sourcesanspro/"
     "SourceSansPro-Regular.ttf", 56),
    ("https://github.com/google/fonts/raw/main/ofl/ptsans/"
     "PTSans-Regular.ttf", 56),
]

# Short phrases that look like overlay text in augmented photos.
PHRASE_BANK = [
    "WELCOME",
    "OPEN 24H",
    "SALE",
    "PUSH",
    "PULL",
    "EXIT",
    "ENTRANCE",
    "NO PARKING",
    "KEEP CLEAR",
    "CAUTION",
    "ROAD WORK",
    "STOP",
    "ONE WAY",
    "HIGHWAY 7",
    "MAIN ST",
    "TOWER 5",
    "BUILDING 12",
    "ROOM 301",
    "FLOOR 3F",
    "OFFICE",
    "RESTAURANT",
    "HOTEL",
    "BANK",
    "PHARMACY",
    "BUS STOP",
    "TICKETS",
    "INFO",
    "RESTROOMS",
    "CAFE",
    "BAR",
    "SHOP",
    "SCHOOL ZONE",
    "PARK",
    "MUSEUM",
    "LIBRARY",
    "POLICE",
    "HOSPITAL",
    "GAS STATION",
    "RECYCLE",
    "WARNING",
]


def download_div2k(target_dir: Path, force: bool = False) -> Path:
    """Download and extract DIV2K train HR. Returns the HR directory."""
    hr_dir = target_dir / "DIV2K_train_HR"
    if hr_dir.exists() and any(hr_dir.glob("*.png")) and not force:
        print(f"[div2k] already extracted at {hr_dir}")
        return hr_dir

    zip_path = target_dir / "DIV2K_train_HR.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1_000_000:
        print(f"[div2k] downloading from {DIV2K_TRAIN_URL} (this may take "
              f"a few minutes)...")
        try:
            urllib.request.urlretrieve(DIV2K_TRAIN_URL, str(zip_path))
        except Exception as e:
            print(f"[div2k] direct download failed: {e}")
            print("[div2k] falling back to gdown mirror")
            try:
                import gdown  # type: ignore
                gdown.download(
                    "https://drive.google.com/uc?id=1TYRd65vK2H1i2lMrl4RdAv"
                    "uZW3cxn9U", str(zip_path), quiet=False
                )
            except Exception as e2:
                raise RuntimeError(
                    f"cannot obtain DIV2K train HR; install gdown or "
                    f"download manually: direct={e}, gdown={e2}"
                ) from e2

    print(f"[div2k] extracting to {target_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    return hr_dir


def collect_fonts(fonts_dir: Path) -> list[Path]:
    """Return a list of TTF/OTF font paths. Tries apt + extra downloads."""
    fonts_dir.mkdir(parents=True, exist_ok=True)

    collected: list[Path] = []

    # 1. system fonts (Colab ships a large /usr/share/fonts tree)
    sys_fonts = []
    for root in [Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]:
        if root.exists():
            sys_fonts.extend(root.rglob("*.ttf"))
            sys_fonts.extend(root.rglob("*.otf"))
    sys_fonts = [p for p in sys_fonts if p.stat().st_size > 5_000]
    print(f"[fonts] {len(sys_fonts)} system fonts found")
    collected.extend(sys_fonts[:12])  # cap to keep variety but bounded

    # 2. extra OFL fonts from google/fonts github
    for url, _ in EXTRA_FONTS:
        fname = url.rsplit("/", 1)[-1].split("%5B")[0].split("?")[0]
        target = fonts_dir / fname
        if target.exists() and target.stat().st_size > 5_000:
            collected.append(target)
            continue
        try:
            urllib.request.urlretrieve(url, str(target))
            if target.stat().st_size > 5_000:
                collected.append(target)
                print(f"[fonts] downloaded {fname}")
            else:
                target.unlink(missing_ok=True)
        except Exception as e:
            print(f"[fonts] could not fetch {fname}: {e}")
            target.unlink(missing_ok=True)

    # fallback: install a few packages quickly if we ended up with almost none
    if len(collected) < 4:
        print("[fonts] too few fonts, installing fonts-dejavu fonts-liberation"
              " via apt")
        try:
            subprocess.run(
                ["apt-get", "install", "-y", "-qq",
                 "fonts-dejavu", "fonts-liberation", "fonts-freefont-ttf"],
                check=False,
            )
        except Exception:
            pass
        collected = list({p for p in collected})
        for root in [Path("/usr/share/fonts/truetype/dejavu"),
                     Path("/usr/share/fonts/truetype/liberation"),
                     Path("/usr/share/fonts/truetype/freefont")]:
            if root.exists():
                collected.extend(list(root.glob("*.ttf"))[:3])

    # de-dup keeping order
    seen = set()
    unique: list[Path] = []
    for p in collected:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if not unique:
        raise RuntimeError("no usable fonts found")
    print(f"[fonts] final font list size = {len(unique)}")
    return unique


def center_crop_or_resize(img: Image.Image, target: int) -> Image.Image:
    w, h = img.size
    # center crop to square
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    if img.mode != "RGB":
        img = img.convert("RGB")
    if side != target:
        img = img.resize((target, target), Image.BICUBIC)
    return img


def render_overlay_text(gt: Image.Image, mask: Image.Image,
                        fonts: list[Path], rng: random.Random) -> None:
    """Render 1-4 text boxes onto a copy kept in-place modifies gt + mask."""
    draw_gt = ImageDraw.Draw(gt)
    draw_mk = ImageDraw.Draw(mask)
    w, h = gt.size

    n_boxes = rng.randint(1, 4)
    for _ in range(n_boxes):
        font_path = rng.choice(fonts)
        phrase = rng.choice(PHRASE_BANK)
        # font size 4-12% of image height
        fs = rng.randint(int(h * 0.04), int(h * 0.12))
        try:
            font = ImageFont.truetype(str(font_path), fs)
        except Exception:
            continue
        # text bbox
        try:
            bbox = draw_gt.textbbox((0, 0), phrase, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = (len(phrase) * fs // 2, fs)
        if text_w <= 0 or text_h <= 0:
            continue
        # random position ensuring text fully within frame
        max_x = max(0, w - text_w - 8)
        max_y = max(0, h - text_h - 8)
        if max_x <= 0 or max_y <= 0:
            continue
        x = rng.randint(4, max_x)
        y = rng.randint(4, max_y)
        # color: pick from white/black/dark blue/red for variety
        color = rng.choice([
            (255, 255, 255),
            (0, 0, 0),
            (220, 30, 30),
            (30, 60, 220),
            (240, 200, 0),
        ])
        # light shadow for contrast on photo background, then main text
        shadow_offset = rng.randint(2, 4)
        draw_gt.text((x + shadow_offset, y + shadow_offset), phrase,
                     fill=tuple(min(255, c // 2) for c in color),
                     font=font)
        draw_gt.text((x, y), phrase, fill=color, font=font)

        # mask: paint a filled rectangle slightly larger than the text bbox
        pad = max(2, fs // 8)
        draw_mk.rectangle(
            [x - bbox[0] - pad, y - bbox[1] - pad,
             x - bbox[0] + text_w + pad, y - bbox[1] + text_h + pad],
            fill=255,
        )


def degrade(img_pil: Image.Image, lr_size: int, hr_size: int,
            rng: random.Random) -> Image.Image:
    """Apply real-world-ish degradation and return an HR-size image."""
    arr = np.asarray(img_pil)  # uint8 HxWx3 RGB

    # 1. bicubic downsample (simulate low-res capture)
    small = cv2.resize(arr, (lr_size, lr_size),
                      interpolation=cv2.INTER_CUBIC)

    # 2. gaussian blur (camera defocus / motion blur approximation)
    sigma = rng.uniform(0.4, 1.6)
    ksize = int(sigma * 6) | 1  # ensure odd
    blurred = cv2.GaussianBlur(small, (ksize, ksize), sigma)

    # 3. additive gaussian noise
    noise_std = rng.uniform(1.0, 12.0)
    noise = np.random.randn(*blurred.shape).astype(np.float32) * noise_std
    noisy = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. JPEG compression artifacts (random quality)
    quality = rng.randint(35, 85)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    else:
        decoded = noisy

    # 5. resize back to hr_size (bicubic) so the dataset returns consistent
    # spatial dimension; the degradation is preserved in the pixel content.
    out = cv2.resize(decoded, (hr_size, hr_size),
                     interpolation=cv2.INTER_CUBIC)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def build_dataset(output_dir: Path, num_images: int, hr_size: int,
                  lr_size: int, seed: int, div2k_cache: Path,
                  fonts_dir: Path) -> None:
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed + 1)

    hr_dir = download_div2k(div2k_cache)
    backgrounds = sorted(hr_dir.glob("*.png"))
    if not backgrounds:
        backgrounds = sorted(hr_dir.glob("*.jpg"))
    if not backgrounds:
        raise RuntimeError(f"no backgrounds found under {hr_dir}")
    print(f"[data] {len(backgrounds)} background images available")

    fonts = collect_fonts(fonts_dir)

    out_hr = output_dir / "HR"
    out_lr = output_dir / "LR"
    out_mk = output_dir / "Mask"
    for d in (out_hr, out_lr, out_mk):
        d.mkdir(parents=True, exist_ok=True)

    written = 0
    while written < num_images:
        bg_path = rng.choice(backgrounds)
        try:
            base = Image.open(bg_path)
        except Exception as e:
            print(f"[data] skip {bg_path}: {e}")
            continue
        gt = center_crop_or_resize(base, hr_size)
        mask = Image.new("L", (hr_size, hr_size), 0)
        render_overlay_text(gt, mask, fonts, rng)
        lr = degrade(gt, lr_size=lr_size, hr_size=hr_size, rng=rng)

        name = f"{written:06d}.png"
        gt.save(out_hr / name)
        lr.save(out_lr / name)
        mask.save(out_mk / name)
        written += 1
        if written % 50 == 0:
            print(f"[data] wrote {written}/{num_images}")

    print(f"[data] done -> {output_dir} ({written} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/content/tadisr_dataset")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--hr-size", type=int, default=1024)
    parser.add_argument("--lr-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--div2k-cache", default="/content/div2k_cache")
    parser.add_argument("--fonts-dir", default="/content/tadisr_fonts")
    args = parser.parse_args()

    # pull NP seed into the global RNG used by degrade()
    np.random.seed(args.seed + 1)
    build_dataset(
        output_dir=Path(args.output_dir),
        num_images=args.num_images,
        hr_size=args.hr_size,
        lr_size=args.lr_size,
        seed=args.seed,
        div2k_cache=Path(args.div2k_cache),
        fonts_dir=Path(args.fonts_dir),
    )


if __name__ == "__main__":
    main()