# Fruit Freshness Classification Web App

This repository contains a Flask-based web application for classifying fruit images as fresh or rotten using three PyTorch models:
- Baseline CNN
- Improved CNN
- ResNet18 transfer learning model

The app loads pretrained weights and provides a simple browser UI to upload an image and compare predictions from all models.

## Repository Structure

- `app.py` — Flask application entrypoint that loads model weights and serves the UI/API.
- `python_script.py` — dataset split utility for dividing an image dataset into `train`, `val`, and `test` folders.
- `templates/index.html` — frontend page for image upload and prediction results.
- `baseline_best.pth` — pretrained weights for the baseline CNN.
- `improved_best.pth` — pretrained weights for the improved CNN.
- `transfer_best.pth` — pretrained weights for the ResNet18 transfer learning model.
- `code.ipynb` — notebook used for model training and experiments.

## Requirements

Install the needed Python packages before running the app:

```bash
pip install flask torch torchvision pillow
```

> Use a Python environment that matches your local setup. GPU support is optional; the app falls back to CPU automatically.

## Running the App

1. Open a terminal in the repository root.
2. Set the Flask entrypoint and run the app:

```bash
set FLASK_APP=app.py
flask run
```

3. Open the displayed local URL in your browser.
4. Upload a fruit image and click `Predict Image`.

## How It Works

The Flask app loads three PyTorch models and applies the same preprocessing transform used during training:
- Resize to `224x224`
- Convert to tensor
- Normalize with ImageNet mean/std

The app returns a JSON response with predictions and confidence scores for each model.

## Dataset Split Utility

Use `python_script.py` to split a labeled image dataset into training, validation, and test sets.

The script expects a source folder structure like:

```text
dataset/
  freshapples/
  freshbanana/
  freshoranges/
  rottenapples/
  rottenbanana/
  rottenoranges/
```

Run it with:

```bash
python python_script.py
```

It creates a `dataset_split/` folder with `train`, `val`, and `test` subfolders.

## Notes

- The class order is fixed and must match training: `freshapples`, `freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`, `rottenoranges`.
- If you change the model architecture or class order, retrain the models and update the saved weights accordingly.

## Troubleshooting

- If the web app fails to start, verify that `baseline_best.pth`, `improved_best.pth`, and `transfer_best.pth` are present in the root directory.
- If predictions are incorrect, confirm that the uploaded image is in RGB format and the model weights match the architecture.
