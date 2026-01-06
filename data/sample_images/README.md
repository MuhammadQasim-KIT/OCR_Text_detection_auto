# Sample Images

This folder contains **example input images** used to demonstrate the OCR pipeline.

The images represent different real-world OCR scenarios:
- Tables with printed and handwritten numbers
- Posters with stylized and sparse text
- Newspaper stock listings
- CAPTCHA-like images (included to demonstrate unreliable OCR detection)

## Notes
- Images are for **demonstration and testing only**
- Large datasets are intentionally not included
- CAPTCHA-style images are expected to be flagged as `__UNRELIABLE`

To run OCR on these images:
python run_ocr.py