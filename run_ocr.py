from pathlib import Path
import cv2
import pandas as pd
import pytesseract

from src.preprocess import (
    preprocess_adaptive,
    preprocess_adaptive_deskew,
    preprocess_otsu,
    preprocess_otsu_deskew,
    preprocess_otsu_table,
    preprocess_otsu_table_deskew,
)
from src.auto_ocr import Pipeline, run_auto_ocr
from src.visualize import draw_boxes

# If Tesseract isn't in PATH, set it here:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INPUT_DIR = Path("data/sample_images")
OUT_TEXT = Path("outputs/text")
OUT_VIZ = Path("outputs/viz")
OUT_METRICS = Path("outputs/metrics")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def ensure_dirs() -> None:
    for d in [OUT_TEXT, OUT_VIZ, OUT_METRICS]:
        d.mkdir(parents=True, exist_ok=True)

def list_images(input_dir: Path):
    return sorted([p for p in input_dir.glob("*") if p.suffix.lower() in SUPPORTED_EXTS])

def main() -> None:
    ensure_dirs()

    image_paths = list_images(INPUT_DIR)
    if not image_paths:
        print(f"No images found in: {INPUT_DIR.resolve()}")
        print("Add at least one image to data/sample_images/ and run again.")
        return

    pipelines = [
        # Tables / newspapers / grids
        Pipeline("otsu_table_psm6", preprocess_otsu_table, 6),
        Pipeline("otsu_table_deskew_psm6", preprocess_otsu_table_deskew, 6),

        # Clean documents
        Pipeline("otsu_psm6", preprocess_otsu, 6),
        Pipeline("otsu_deskew_psm6", preprocess_otsu_deskew, 6),
        Pipeline("otsu_psm4", preprocess_otsu, 4),

        # Posters / sparse layouts
        Pipeline("adapt_psm11", preprocess_adaptive, 11),
        Pipeline("adapt_deskew_psm11", preprocess_adaptive_deskew, 11),

        # Single line attempt
        Pipeline("adapt_psm7", preprocess_adaptive, 7),
    ]

    rows = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read image: {img_path.name}")
            continue

        best_name, best_pre, words, best_score = run_auto_ocr(img, pipelines)

        # Save best preprocessed image for debugging
        cv2.imwrite(str(OUT_VIZ / f"{img_path.stem}_{best_name}_pre.png"), best_pre)

        extracted_text = " ".join([w["text"] for w in words if w["text"].strip()])
        (OUT_TEXT / f"{img_path.stem}.txt").write_text(extracted_text, encoding="utf-8")

        viz = draw_boxes(img.copy(), words)
        cv2.imwrite(str(OUT_VIZ / f"{img_path.stem}_ocr_boxes.png"), viz)

        for w in words:
            rows.append({
                "image": img_path.name,
                "pipeline": best_name,
                "score": best_score,
                "text": w["text"],
                "conf": w["conf"],
                "x": w["x"], "y": w["y"], "w": w["w"], "h": w["h"],
            })

        print(f"Processed: {img_path.name} | pipeline={best_name} | words_kept={len(words)}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_METRICS / "ocr_words.csv", index=False)
        print(f"\nSaved per-word metrics: {OUT_METRICS / 'ocr_words.csv'}")
    else:
        print("\nNo OCR words passed the filters (try lowering the confidence threshold in src/ocr.py).")

    print(f"Outputs folder: {Path('outputs').resolve()}")

if __name__ == "__main__":
    main()
