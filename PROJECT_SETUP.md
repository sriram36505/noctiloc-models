# NoctiLoc Models - Project Setup Guide

## Overview
This is an advanced implementation of YOLOv8 for **Nighttime Vehicle Detection** using:
- Temporal Sequence Processing with ConvLSTM
- Headlight Attention Masking
- CUDA-optimized training on GPU

## Project Structure

```
noctiloc-models/
├── notebooks/                  # Jupyter notebooks for training & inference
│   ├── 1_YOLOv8_Training.ipynb
│   └── 2_Temporal_YOLO_Model.ipynb
├── src/                        # Source code
│   ├── training.py            # Training utilities
│   ├── data_utils.py          # Data processing
│   ├── model.py               # Model definitions  
│   └── inference.py           # Inference scripts
├── models/                     # Trained model weights
│   ├── yolov8n_nighttime.pt
│   ├── yolov8s_nighttime.pt
│   └── temporal_yolo.pth
├── data/                       # Dataset annotations
├── results/                    # Training outputs
├── requirements.txt            # Dependencies
├── LICENSE                     # MIT License
└── README.md                   # Main documentation
```

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/sriram36505/noctiloc-models.git
cd noctiloc-models
pip install -r requirements.txt
```

### 2. Training
```bash
python src/training.py --model yolov8s --epochs 50 --batch-size 32
```

### 3. Inference
```bash
python src/inference.py --model models/yolov8s_nighttime.pt --image test.jpg
```

## Dataset Information
- **Total Images**: 2919 nighttime vehicle images
- **Training**: 2335 images (80%)
- **Validation**: 584 images (20%)
- **Image Size**: 1280 × 1024 pixels
- **Annotation Format**: YOLO format (normalized coordinates)
- **Single Class**: Vehicle

## Model Performance

### YOLOv8 Nano
- mAP@0.5: 0.41
- Parameters: 3.0M
- Speed: ~45 FPS (GPU)

### YOLOv8 Small
- mAP@0.5: 0.45
- Parameters: 11.2M
- Speed: ~30 FPS (GPU)

### Temporal YOLOv8
- Processes 5-frame sequences
- Improved temporal consistency
- Better performance on motion blur scenarios

## Key Features
✅ YOLOv8 architecture with temporal processing
✅ ConvLSTM for sequence understanding
✅ Headlight detection with attention masking
✅ Data augmentation optimized for nighttime
✅ CUDA acceleration support
✅ Multi-scale detection
✅ Comprehensive evaluation metrics

## Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA 12.1+ (for GPU acceleration)
- 8GB+ GPU VRAM recommended

## Usage Examples

### Training Custom Dataset
```python
from src.training import YOLOv8Trainer

trainer = YOLOv8Trainer()
results = trainer.train(
    data_yaml='path/to/data.yaml',
    epochs=100,
    batch_size=32,
    model='yolov8s'
)
```

### Inference on Image
```python
from src.inference import NightDetector

detector = NightDetector('models/yolov8s_nighttime.pt')
results = detector.detect('night_image.jpg')
detector.visualize_results(results)
```

## Citation
If you use these models in your research, please cite:
```bibtex
@software{noctilocmodels2024,
  title={NoctiLoc: Nighttime Vehicle Detection Models},
  author={Sriram, R.},
  year={2024},
  url={https://github.com/sriram36505/noctiloc-models}
}
```

## License
MIT License - See LICENSE file for details

## Contributing
Contributions are welcome! Please feel free to submit pull requests.

## Support
For issues and questions, please open an issue on GitHub.

---
**Last Updated**: January 2026
**Maintained By**: Sriram R (@sriram36505)
