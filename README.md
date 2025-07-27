# Multimodal Authentication and Product Recommendation System

## Overview
This project implements a multimodal authentication system that combines facial recognition and voice verification to ensure secure access to a product recommendation engine. The system is designed to process tabular, image, and audio data, and integrates machine learning models for authentication and recommendation.

---

## Key Features
1. **Tabular Data Processing**:
   - Merges customer social profiles and transaction data.
   - Handles missing values and normalizes formats.
   - Performs one-hot encoding for categorical features.

2. **Image Processing**:
   - Extracts facial features using embeddings and histograms.
   - Applies augmentations such as rotation and flipping.
   - Supports facial recognition for authentication.

3. **Audio Processing**:
   - Extracts audio features like MFCCs and spectral roll-off.
   - Applies augmentations such as pitch shift and time stretch.
   - Supports voice verification for authentication.

4. **Machine Learning Models**:
   - Facial Recognition Model.
   - Voiceprint Verification Model.
   - Product Recommendation Model.

5. **Command-Line Interface**:
   - Simulates authentication and recommendation workflows.
   - Handles unauthorized attempts and full transaction simulations.

---

## Core Components

### 1. Tabular Data Processing
**File**: `scripts/merge_tabular_data.py`
- **Functions**:
  - `load_data`: Loads customer profiles and transactions.
  - `normalize_customer_ids`: Standardizes customer IDs for merging.
  - `merge_data`: Merges datasets on normalized IDs.
  - `handle_missing_data`: Fills missing values with medians or "Unknown."
  - `normalize_formats`: Standardizes date formats.
  - `one_hot_encode`: Encodes categorical features.

### 2. Image Processing
**File**: `scripts/image_features.py`
- **Functions**:
  - `augment_image`: Applies augmentations like rotation and flipping.
  - `extract_features`: Extracts embeddings and histograms for facial recognition.

### 3. Audio Processing
**File**: `scripts/audio_processing.py`
- **Functions**:
  - `extract_features`: Extracts MFCCs and other audio features.
  - `augment_audio`: Applies augmentations like pitch shift and time stretch.
  - `process_audio`: Processes audio files and extracts features.
  - `train_model`: Trains the voice verification model.
- **Class**:
  - `VoiceVerifier`: Loads the trained model and verifies audio samples.

### 4. Command-Line Interface
**File**: `scripts/demo_auth_cli.py`
- **Functions**:
  - `extract_face`: Detects and extracts facial features from an image.
  - `extract_voice_features`: Extracts audio features for verification.
  - `predict_face`: Predicts face authentication status.
  - `verify_voice`: Verifies voice authentication status.
  - `main`: Simulates the multimodal authentication and recommendation workflow.

---

## Workflow
1. **Data Preparation**:
   - Merge tabular data and preprocess image and audio datasets.
   - Extract and save features into CSV files.

2. **Model Training**:
   - Train facial recognition, voice verification, and product recommendation models.

3. **System Simulation**:
   - Simulate authentication and recommendation workflows via CLI.

---

## Deliverables
- **Datasets**:
  - Merged dataset with feature engineering.
  - `image_features.csv` and `audio_features.csv`.

- **Scripts**:
  - Python scripts for data processing, feature extraction, and model training.
  - CLI for system simulation.

- **Models**:
  - Pre-trained models for facial recognition, voice verification, and product recommendation.

- **Documentation**:
  - Detailed report and system simulation video.

---
