# OCR Text Detection with Automatic Pipeline Selection

This project implements a robust OCR (Optical Character Recognition) system using **Tesseract OCR** and **OpenCV**, designed to work on heterogeneous document images such as tables, newspapers, posters, and noisy scans.

Instead of relying on a single OCR configuration, the system automatically selects the best preprocessing + OCR pipeline **per image** based on confidence-driven heuristics.

---

## Key Features

- Multiple OCR preprocessing pipelines:
  - Adaptive thresholding
  - Otsu thresholding
  - Deskew (rotation correction)
  - Table/grid line removal
- Automatic pipeline selection per image
- Word-level OCR with bounding boxes and confidence scores
- Automatic detection of unreliable OCR cases (e.g. CAPTCHA-like images)
- Debug visualizations of preprocessing and OCR results
- CSV export with per-word OCR metrics

---

## Project Structure

```
ocr-text-detection/
├── data/
│   └── sample_images/        # input images
├── outputs/
│   ├── text/                # extracted text (.txt)
│   ├── viz/                 # visualizations (preprocessed + boxes)
│   └── metrics/             # CSV metrics
├── src/
│   ├── preprocess.py
│   ├── ocr.py
│   ├── auto_ocr.py
│   └── visualize.py
└── run_ocr.py
```

---

## How It Works

For each input image, the system:

1. Tries multiple OCR pipelines (preprocessing + PSM).
2. Runs OCR for each pipeline.
3. Scores results using:
   - mean confidence
   - number of detected words
   - word-length heuristics
4. Selects the best-performing pipeline automatically.
5. Flags images as `__UNRELIABLE` if OCR confidence is too low.

---

## Running the Pipeline

```powershell
python run_ocr.py
```

Input images must be placed in:

```
data/sample_images/
```

---

## Example Results (Actual Run)

Console output from a real run:

```
Processed: captcha2.jpg | pipeline=adapt_deskew_psm11__UNRELIABLE | words_kept=1
Processed: numbers_gs150.jpg | pipeline=otsu_psm4 | words_kept=95
Processed: plaid_bw200.jpg | pipeline=otsu_psm4 | words_kept=20
Processed: stock_gs200q25.jpg | pipeline=adapt_psm11 | words_kept=200
```

---

## Visual Results

### CAPTCHA-like Image (Automatically Flagged)

- Input: `data/sample_images/captcha2.jpg`
- Selected pipeline: `adapt_deskew_psm11__UNRELIABLE`

Preprocessed image:
```
outputs/viz/captcha2_adapt_deskew_psm11__UNRELIABLE_pre.png
```

OCR bounding boxes:
```
outputs/viz/captcha2_ocr_boxes.png
```

---

### Handwritten + Printed Table

- Input: `data/sample_images/numbers_gs150.jpg`
- Selected pipeline: `otsu_psm4`
- Words detected: 95

Preprocessed image:
```
outputs/viz/numbers_gs150_otsu_psm4_pre.png
```

OCR bounding boxes:
```
outputs/viz/numbers_gs150_ocr_boxes.png
```

---

### Poster / Stylized Text

- Input: `data/sample_images/plaid_bw200.jpg`
- Selected pipeline: `otsu_psm4`
- Words detected: 20

Preprocessed image:
```
outputs/viz/plaid_bw200_otsu_psm4_pre.png
```

OCR bounding boxes:
```
outputs/viz/plaid_bw200_ocr_boxes.png
```

---

### Newspaper Stock Listings

- Input: `data/sample_images/stock_gs200q25.jpg`
- Selected pipeline: `adapt_psm11`
- Words detected: 200

Preprocessed image:
```
outputs/viz/stock_gs200q25_adapt_psm11_pre.png
```

OCR bounding boxes:
```
outputs/viz/stock_gs200q25_ocr_boxes.png
```

---

## Generated Outputs

- Extracted text:
```
outputs/text/<image_name>.txt
```

- OCR metrics (CSV):
```
outputs/metrics/ocr_words.csv
```

The CSV contains:
- image name
- selected pipeline
- OCR score
- detected word
- confidence
- bounding box coordinates

---

## Known Limitations

- CAPTCHA-like images are intentionally flagged as unreliable.
- Handwritten text is partially supported.
- Decorative logos may not be recognized correctly.

These limitations are explicitly handled by the system.

---

## Portfolio Value

This project demonstrates:
- OCR system design
- classical computer vision preprocessing
- automatic decision-making without ground truth
- explainability via visual debugging
- robust handling of real-world document images
