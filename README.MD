# YOLOv12 Face Detection

A face detection implementation using YOLOv12 models, evaluated on the WiderFace dataset. This project provides training scripts, evaluation framework, and pre-trained models for face detection tasks.

## Features

- 🚀 **State-of-the-art Performance**: YOLOv12 models with excellent accuracy on WiderFace
- 📊 **Comprehensive Evaluation**: WiderFace evaluation protocol implementation
- 🔧 **Easy Training**: Simple training script with customizable parameters
- 📦 **Multiple Model Sizes**: From nano to extra-large variants
- 🎯 **Ready-to-use**: Pre-trained weights and ONNX format available

## Results on WiderFace Validation Set

| Model | Easy | Medium | Hard | Weights |
|-------|------|--------|------|---------|
| YOLOv12n | 0.9400 | 0.9188 | 0.7786 | [PyTorch](link-to-be-added) \| [ONNX](link-to-be-added) |
| YOLOv12s | 0.9529 | 0.9376 | 0.8205 | [PyTorch](link-to-be-added) \| [ONNX](link-to-be-added) |
| YOLOv12m | 0.9590 | 0.9466 | 0.8422 | [PyTorch](link-to-be-added) \| [ONNX](link-to-be-added) |
| YOLOv12l | 0.9620 | 0.9508 | 0.8473 | [PyTorch](link-to-be-added) \| [ONNX](link-to-be-added) |
| YOLOv12x | 0.9630 | 0.9528 | 0.8549 | [PyTorch](link-to-be-added) \| [ONNX](link-to-be-added) |

*Evaluation performed on WiderFace validation set using standard evaluation protocol.*

## Installation

### Requirements

```bash
pip install ultralytics
```

### Clone Repository

```bash
git clone https://github.com/praysimanjuntak/yolov12face.git
cd yolov12face
```

## Quick Start

### Inference

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('path/to/yolov12n_face.pt')

# Run inference
results = model('examples/face.jpg')

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        print(f"Face detected with confidence: {box.conf}")
```

### Training

```bash
python train.py
```

The training script will:
- Load YOLOv12n as base model
- Train on your dataset (configured in `data/dataset.yaml`)
- Run for 100 epochs with image size 640

### Evaluation on WiderFace

```bash
# Download WiderFace validation set and ground truth
# Then run evaluation:

python evaluation_yolo.py \
    --model path/to/your/model.pt \
    --images path/to/widerface/val/images \
    --gt path/to/widerface/ground_truth \
    --output ./predictions
```

## Dataset Structure

For training, organize your dataset as follows:

```
data/
├── dataset.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Example `dataset.yaml`:

```yaml
train: data/images/train
val: data/images/val
nc: 1
names: ['face']
```

## Model Architecture

This implementation uses YOLOv12 architecture optimized for face detection:

- **Input Size**: 640×640
- **Classes**: 1 (face)
- **Anchor-free**: Modern YOLO architecture without predefined anchors
- **Multi-scale**: Detects faces at various scales

## Evaluation Framework

The evaluation framework (`evaluation_base.py` and `evaluation_yolo.py`) provides:

- **WiderFace Protocol**: Standard evaluation metrics
- **Modular Design**: Easy to extend for other models
- **Detailed Results**: Easy, Medium, Hard subset evaluation
- **IoU Threshold**: Configurable evaluation criteria

### Custom Model Evaluation

To evaluate your own model, inherit from `BaseWiderFaceEvaluator`:

```python
from evaluation_base import BaseWiderFaceEvaluator

class CustomEvaluator(BaseWiderFaceEvaluator):
    def detect_faces(self, image_path):
        # Implement your detection logic
        # Return: [[x1, y1, width, height, confidence], ...]
        pass
```

## Examples

### Basic Detection

```python
from ultralytics import YOLO
import cv2

model = YOLO('yolov12n_face.pt')
image = cv2.imread('examples/face.jpg')

results = model(image)
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    for box in boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), (0, 255, 0), 2)

cv2.imwrite('result.jpg', image)
```

### Batch Processing

```python
import os
from ultralytics import YOLO

model = YOLO('yolov12n_face.pt')
image_dir = 'path/to/images'

for img_file in os.listdir(image_dir):
    if img_file.lower().endswith(('.jpg', '.png')):
        results = model(os.path.join(image_dir, img_file))
        # Process results...
```

## Performance Tips

1. **Image Size**: Use 640×640 for best accuracy/speed balance
2. **Confidence Threshold**: Adjust based on your use case (default: 0.001)
3. **NMS Threshold**: Fine-tune for overlapping faces
4. **Batch Size**: Increase for faster processing of multiple images

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv12 implementation
- [WiderFace](http://shuoyang1213.me/WIDERFACE/) dataset creators
- Original WiderFace evaluation code contributors

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{yolov12face2025,
  title={YOLOv12 Face Detection},
  author={Pray Apostel},
  year={2024},
  url={https://github.com/praysimanjuntak/yolov12face}
}
```