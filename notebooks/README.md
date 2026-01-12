# Jupyter Notebooks

This directory contains Jupyter notebooks for training and evaluating YOLOv8-based nighttime vehicle detection models.

## Notebooks

### APP1-3.ipynb
YOLOv8 model training and temporal sequence processing with ConvLSTM layers for improved nighttime vehicle detection accuracy. This notebook covers:
- Dataset preparation and loading
- YOLOv8 model configuration
- ConvLSTM integration for temporal processing
- Training pipeline and hyperparameter tuning
- Performance evaluation and metrics

### APP2-3.ipynb  
Headlight attention masking and advanced post-processing techniques for nighttime detection. This notebook includes:
- Headlight region detection and masking
- Attention mechanism implementation
- Detection post-processing
- Visualization of detection results
- Performance comparison with baseline models

## Usage

1. Ensure all dependencies from `requirements.txt` are installed
2. Prepare your dataset in the appropriate directory
3. Run notebooks in order: APP1-3.ipynb first, then APP2-3.ipynb
4. Modify dataset paths and hyperparameters as needed for your environment

## Dataset

Supported formats:
- YOLOv8 dataset format (images + annotations)
- COCO format
- VOC format

## Results

Expected performance metrics:
- mAP (mean Average Precision): 0.41-0.45 at night
- FPS: 25-30 frames per second
- Detection confidence threshold: 0.5
