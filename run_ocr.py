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

def rescale_words(words, scale: float):
    """Map word boxes from preprocessed image coordinates back to original image coordinates."""
    if scale == 1.0:
        return words
    out = []
    for w in words:
        ww = dict(w)
        ww["x"] = int(w["x"] / scale)
        ww["y"] = int(w["y"] / scale)
        ww["w"] = int(w["w"] / scale)
        ww["h"] = int(w["h"] / scale)
        out.append(ww)
    return out

def main() -> None:
    ensure_dirs()

    image_paths = list_images(INPUT_DIR)
    if not image_paths:
        print(f"No images found in: {INPUT_DIR.resolve()}")
        print("Add at least one image to data/sample_images/ and run again.")
        return

    # IMPORTANT: scale values must match resize factors in preprocess.py
    pipelines = [
        # Tables / newspapers / grids
        Pipeline("otsu_table_psm6", preprocess_otsu_table, 6, scale=2.5, can_map_to_original=True),
        Pipeline("otsu_table_deskew_psm6", preprocess_otsu_table_deskew, 6, scale=2.5, can_map_to_original=False),

        # Clean documents
        Pipeline("otsu_psm6", preprocess_otsu, 6, scale=2.5, can_map_to_original=True),
        Pipeline("otsu_deskew_psm6", preprocess_otsu_deskew, 6, scale=2.5, can_map_to_original=False),
        Pipeline("otsu_psm4", preprocess_otsu, 4, scale=2.5, can_map_to_original=True),

        # Posters / sparse layouts
        Pipeline("adapt_psm11", preprocess_adaptive, 11, scale=2.0, can_map_to_original=True),
        Pipeline("adapt_deskew_psm11", preprocess_adaptive_deskew, 11, scale=2.0, can_map_to_original=False),

        # Single line attempt
        Pipeline("adapt_psm7", preprocess_adaptive, 7, scale=2.0, can_map_to_original=True),
    ]

    rows = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not read image: {img_path.name}")
            continue

        best_p, best_pre, words, best_score = run_auto_ocr(img, pipelines)

        # 1) Save best preprocessed image for debugging
        cv2.imwrite(str(OUT_VIZ / f"{img_path.stem}_{best_p.name}_pre.png"), best_pre)

        # 2) Always save OCR boxes on the PREPROCESSED image (coordinates always match here)
        pre_bgr = cv2.cvtColor(best_pre, cv2.COLOR_GRAY2BGR) if len(best_pre.shape) == 2 else best_pre.copy()
        viz_pre = draw_boxes(pre_bgr, words)
        cv2.imwrite(str(OUT_VIZ / f"{img_path.stem}_ocr_boxes_pre.png"), viz_pre)

        # 3) Save OCR boxes on the ORIGINAL image (only if mapping is valid)
        if best_p.can_map_to_original:
            words_for_original = rescale_words(words, best_p.scale)
            viz_orig = draw_boxes(img.copy(), words_for_original)
            cv2.imwrite(str(OUT_VIZ / f"{img_path.stem}_ocr_boxes.png"), viz_orig)
        else:
            # Still write an image so user isn't confused:
            # we reuse the preprocessed visualization name pattern.
            cv2.imwrite(str(OUT_VIZ / f"{img_path.stem}_ocr_boxes.png"), viz_pre)

        # 4) Save extracted text
        extracted_text = " ".join([w["text"] for w in words if w["text"].strip()])
        (OUT_TEXT / f"{img_path.stem}.txt").write_text(extracted_text, encoding="utf-8")

        # 5) Save metrics rows
        for w in words:
            rows.append({
                "image": img_path.name,
                "pipeline": best_p.name,
                "score": best_score,
                "text": w["text"],
                "conf": w["conf"],
                "x": w["x"], "y": w["y"], "w": w["w"], "h": w["h"],
            })

        print(f"Processed: {img_path.name} | pipeline={best_p.name} | words_kept={len(words)}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_METRICS / "ocr_words.csv", index=False)
        print(f"\nSaved per-word metrics: {OUT_METRICS / 'ocr_words.csv'}")
    else:
        print("\nNo OCR words passed the filters (try lowering the confidence threshold in src/ocr.py).")

    print(f"Outputs folder: {Path('outputs').resolve()}")
    print("Note: 'outputs/viz/*_ocr_boxes_pre.png' is always reliable for visualization.")

if __name__ == "__main__":
    main()
