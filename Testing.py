
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageOps

# ——— parametry docelowe ———
TARGET_SIZE: Tuple[int, int] = (492, 633)    # pix (szer., wys.)
TARGET_DPI:  Tuple[int, int] = (300, 300)
MAX_FILE_SIZE = 2 * 1024 * 1024               # 2 MB
ALLOWED_FORMATS = {"jpg", "jpeg", "png", "bmp", "gif"}

# ——— ścieżki wej./wyj. ———
IN_PATH  = Path(r"C:\Users\marci\Desktop\download.jpg")
OUT_PATH = Path(r"C:\Users\marci\Desktop\download_id1.jpg")


def central_crop(img: Image.Image, ratio: float) -> Image.Image:
    """Kadruje środek obrazu do zadanego stosunku szer./wys."""
    w, h = img.size
    if w / h > ratio:             # obcinamy boki
        new_w = int(h * ratio)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:                         # obcinamy górę/dół
        new_h = int(w / ratio)
        top = (h - new_h) // 2
        box = (0, top, w, top + new_h)
    return img.crop(box)


def save_under_size(img: Image.Image, out_path: Path) -> None:
    """Zapis z DPI 300 i gwarancją ≤ 2 MB (dla JPG redukuje quality)."""
    fmt = out_path.suffix.lower().lstrip(".")
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(f"Niedozwolony format: {fmt}")

    params = {"dpi": TARGET_DPI}

    if fmt in {"jpg", "jpeg"}:
        for q in range(95, 10, -5):
            tmp = out_path.with_suffix(".tmp")
            img.save(tmp, format="JPEG", quality=q,
                     optimize=True, progressive=True, **params)
            if tmp.stat().st_size <= MAX_FILE_SIZE:
                tmp.rename(out_path)
                break
        else:
            raise RuntimeError("Nie udało się zejść poniżej 2 MB.")
    else:
        img.save(out_path, format=fmt.upper(), **params)
        if out_path.stat().st_size > MAX_FILE_SIZE:
            raise RuntimeError("Plik > 2 MB – użyj JPG.")


def process_id_photo(in_path: Path, out_path: Path) -> None:
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    with Image.open(in_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")

        # kadrujemy do proporcji 236 / 297
        ratio = TARGET_SIZE[0] / TARGET_SIZE[1]
        img = central_crop(img, ratio)

        # dokładne 236 × 297 px
        img = ImageOps.fit(img, TARGET_SIZE, method=Image.LANCZOS)

        save_under_size(img, out_path)

    print(f"✓ Gotowe: {out_path}  ({out_path.stat().st_size/1024:.1f} kB)")


if __name__ == "__main__":
    process_id_photo(IN_PATH, OUT_PATH)
