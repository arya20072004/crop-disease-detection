# Crop Disease Detection — Project Context

## Repository overview

This repository implements an end-to-end crop disease detection and early warning system. It combines: 
- a computer vision pipeline for leaf disease classification,
- a weather/time-series model for forecasting yield and disease risk,
- a fusion model that combines visual and weather features,
- a Python API backend,
- a Streamlit analytics/dashboard UI,
- a Flutter mobile application.

The repo appears to be organized as a research/deployment prototype with both model training assets and deployment assets.

---

## High-level architecture

1. **CNN disease classifier**
   - Implemented in `api.py` and trained using files under `increment_1_cnn/`.
   - Uses `EfficientNet-B4` as backbone and exports embeddings, predictions, and Grad-CAM support.
   - Model weights: `best_model_v2.pth`, with legacy weights in `increment_1_cnn/best_model.pth`.

2. **LSTM weather model**
   - Also loaded in `api.py` from `best_lstm.pth`.
   - Performs time-series forecasting on weather features and predicts disease risk patterns over future windows.

3. **Fusion model**
   - Loaded in `api.py` from `best_fusion.pth`.
   - Combines CNN embeddings, weather-context embeddings, and metadata to produce a composite prediction.

4. **Backend/API**
   - `api.py` is a FastAPI application.
   - Supports image upload and inference, integrates disease classification, weather/LSTM prediction, and fusion outputs.
   - Uses local JSON config files: `class_names.json`, `lstm_config.json`, and `fusion_config.json`.

5. **Dashboard/UI**
   - `dashboard_clean.py` is the primary Streamlit dashboard with a polished visual design.
   - `dashboard.py` is an alternate or previous dashboard version.
   - The dashboard uses static/synthetic data generation for demo charts and analytics.

6. **Flutter mobile app**
   - Located in `crop_disease_ews/`.
   - Standard Flutter project structure with `lib/`, `android/`, `ios/`, `windows/`, `linux/`, `web/`, and `pubspec.yaml`.
   - Appears intended for mobile deployment of the disease detection system.

7. **Notebook experiments**
   - `increment_1_cnn/` contains Jupyter notebooks for CNN training and inference.
   - Increment directories also include model assets and dataset folders.

---

## Root-level files and directories

- `README.md`
  - Project documentation and usage instructions.

- `api.py`
  - Main Python backend.
  - Loads CNN, LSTM, and fusion models.
  - Uses FastAPI, Torch, OpenCV, pandas, and PIL.
  - Includes model definitions for:
    - `CropDiseaseClassifier`
    - `WeatherLSTM`
    - `CrossAttentionFusion`

- `dashboard_clean.py`
  - Streamlit dashboard with custom CSS styling and Plotly visualization.
  - Generates demo risk charts, overview cards, and interactive UI sections.

- `dashboard.py`
  - Another Streamlit interface file.
  - Unknown exact differences without deeper review, but likely older or alternate version.

- `best_fusion.pth`
  - Trained fusion model checkpoint.

- `best_lstm.pth`
  - Trained weather/LSTM model checkpoint.

- `best_model_v2.pth`
  - Trained CNN disease classification checkpoint used by `api.py`.

- `class_names.json`
  - Label mapping for disease classes.

- `fusion_config.json`
  - Configuration for fusion metadata and crop/growth stage embeddings.

- `lstm_config.json`
  - LSTM input and forecast configuration.

- `dcws_scores.csv`
  - Disease/crop/weather scoring dataset for analytics or risk computation.

- `weather_features.csv`
  - Weather features used by the LSTM model.

- `yield_model.json`
  - Likely configuration or model metadata for yield prediction.

- `increment_1_cnn/`
  - Contains training notebooks, original model checkpoints, ONNX exports, dataset references, and an API copy.

- `crop_disease_ews/`
  - Flutter project for mobile app deployment.

---

## Inference & deployment files

### `api.py` details

- Uses `torch.device('cuda' if available else 'cpu')`.
- Preprocesses images with `albumentations` to `380x380`, normalize to ImageNet stats, and convert to tensors.
- Provides:
  - disease classification via CNN,
  - Grad-CAM heatmap generation,
  - weather sequence encoding through LSTM,
  - fusion prediction using cross-attention.
- Model dimensions observed in code:
  - `CNN_DIM = 512`
  - `WEATHER_DIM = 256`
  - `META_DIM` from fusion config
  - `FUSION_INPUT_DIM = CNN_DIM + WEATHER_DIM + META_DIM`

### `dashboard_clean.py` details

- Uses Streamlit with custom CSS for a dark theme and modern dashboard styling.
- Builds a dashboard with:
  - KPI cards,
  - alert badges,
  - Plotly charts,
  - synthetic data generation for risk, disease distribution, crop metrics, and feature correlations.
- This file is likely the main UI for presentation/demo use.

---

## Flutter app context

- `crop_disease_ews/README.md`
  - Contains generic Flutter starter guidance.
  - Indicates the mobile app is a typical Flutter project scaffold.

- `crop_disease_ews/lib/`
  - Main Dart source.
  - Likely contains screens, services, providers, and widgets for crop/disease detection.

- `crop_disease_ews/pubspec.yaml`
  - Flutter dependencies and assets.

- `crop_disease_ews/android/`, `ios/`, `linux/`, `macos/`, `windows/`, `web/`
  - Platform-specific build and runtime files.

---

## Experimental and training resources

- `increment_1_cnn/increment_1_cnn.ipynb`
  - Jupyter notebook for CNN training.

- `increment_1_cnn/api.py`
  - A copy or variant of the inference API for the CNN experiment.

- `increment_1_cnn/leafseg.py`
  - Script for leaf segmentation, removing background noise (soil, sky, pots, hands) using HSV masking and GrabCut to isolate the leaf for better disease detection.

- `increment_1_cnn/image_quality.py`
  - Module for assessing image quality, checking for blur, brightness, contrast, and resolution to ensure images are suitable for accurate inference.

- `increment_1_cnn/crop_disease_cnn.onnx` and `crop_disease_cnn_v2.onnx`
  - ONNX exports for deployment compatibility.

- `increment_1_cnn/PlantDoc-Dataset/`
  - Field image dataset used for domain adaptation.

- `increment_1_cnn/plantvillage dataset/`
  - PlantVillage dataset for controlled disease classification.

---

## Usage notes

- The project is currently set up for local Python deployment via FastAPI and Streamlit.
- The Flutter app is a separate front-end path for mobile use.
- The key ML models are checkpoint files in the root of the repository.
- The `api.py` backend is the central integration point for models and inference.

---

## Recommended next steps for a developer

1. Review `api.py` and confirm the FastAPI endpoints.
2. Explore `dashboard_clean.py` for current UI behavior and data flow.
3. Inspect `crop_disease_ews/lib/` for mobile app integration details.
4. Validate dataset paths and any missing dataset files for training.
5. Check whether `dashboard.py` is still required or can be archived.

---

## Notes

- This repository contains both production-style deployment assets and research artifacts.
- The `project_context.md` file is intentionally designed to capture the full repository layout, major components, and usage patterns.
