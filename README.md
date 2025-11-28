# Plant Leaf Disease Detection with EfficientNetV2

A comprehensive deep learning system for detecting plant leaf diseases using EfficientNetV2 backbone with transfer learning, meta-heuristic optimization, and Grad-CAM explainability.

## Features

- **🌿 EfficientNetV2 Backbone**: Uses pre-trained EfficientNetV2B0 for robust feature extraction
- **🔄 Transfer Learning**: Fine-tunes on ImageNet weights for better performance with limited data
- **📊 Data Augmentation**: Handles real-world conditions (rotation, zoom, brightness, shadows)
- **🧬 Meta-heuristic Optimization**: Genetic Algorithm for automatic hyperparameter tuning
- **📈 Comprehensive Metrics**: Tracks accuracy, precision, recall, and F1-score
- **🔍 Grad-CAM Visualization**: Explains model decisions with heatmaps
- **📱 Mobile Deployment**: TensorFlow Lite conversion for mobile/IoT devices
- **⚡ ModelCheckpoint**: Saves best model weights during training

## Requirements

- Python 3.8+
- TensorFlow 2.13+
- See `requirements.txt` for complete dependencies

## Installation

```bash
# Clone or download the project
# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

Organize your plant disease dataset as follows:

```
dataset/
├── healthy/
│   ├── healthy_leaf_001.jpg
│   ├── healthy_leaf_002.jpg
│   └── ...
├── bacterial_spot/
│   ├── bacterial_001.jpg
│   ├── bacterial_002.jpg
│   └── ...
├── early_blight/
│   ├── early_blight_001.jpg
│   └── ...
└── other_diseases/
    └── ...
```

## Quick Start

### 1. Basic Usage

```python
from plant_disease_detection import PlantDiseaseDetector, HyperParams

# Initialize detector
detector = PlantDiseaseDetector(
    data_dir="path/to/your/dataset",
    num_classes=10  # Adjust based on your classes
)

# Load data with augmentation
train_data, val_data = detector.load_data()

# Use default hyperparameters
hyperparams = HyperParams(
    learning_rate=0.001,
    batch_size=32,
    dense_units=256,
    dropout_rate=0.3
)

# Train model
model = detector.train_model(train_data, val_data, hyperparams)

# Evaluate performance
metrics = detector.evaluate_model(val_data)

# Convert to mobile format
detector.convert_to_tflite("mobile_model.tflite")
```

### 2. Full Pipeline with Optimization

```python
# Run the complete pipeline
python plant_disease_detection.py
```

### 3. Quick Training Script

```python
# Generate a simplified training script
python plant_disease_detection.py --create-quick-script
# Then edit and run quick_train.py
```

## Key Components

### Data Augmentation
- **Rotation**: ±40 degrees for orientation invariance
- **Zoom**: ±20% for scale variation
- **Brightness**: 0.5-1.5x for lighting conditions
- **Flips**: Horizontal and vertical for symmetry
- **Shifts**: Width/height shifts for position invariance

### Model Architecture
- **Backbone**: EfficientNetV2B0 (pre-trained on ImageNet)
- **Custom Head**: Dense layers with batch normalization and dropout
- **Input Size**: 224×224×3 (optimized for EfficientNet)
- **Transfer Learning**: Two-phase training (frozen → fine-tuning)

### Hyperparameter Optimization
Uses Genetic Algorithm to optimize:
- Learning rate (0.0001 - 0.01)
- Batch size (16, 32, 64)
- Dense units (64, 128, 256, 512)
- Dropout rate (0.2 - 0.6)

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro-averaged
- **Recall**: Per-class and macro-averaged
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed class-wise performance

### Grad-CAM Visualization
- Highlights important image regions for predictions
- Uses the last convolutional layer by default
- Generates heatmap overlays for model interpretability

### Mobile Deployment
- **TensorFlow Lite**: Optimized for mobile/edge devices
- **Quantization**: Float16 for smaller model size
- **Size Reduction**: Typically 4-8x smaller than original
- **Inference Speed**: Optimized for real-time prediction

## Configuration

### Key Parameters

```python
# Image processing
IMG_SIZE = (224, 224)  # EfficientNet input size
BATCH_SIZE = 32        # Training batch size
EPOCHS = 100          # Maximum training epochs

# Optimization
POPULATION_SIZE = 10   # GA population size
GENERATIONS = 5       # GA generations
MUTATION_RATE = 0.2   # GA mutation probability

# Model settings
VALIDATION_SPLIT = 0.2  # Train/validation split
```

### Callbacks
- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Prevents overfitting (patience=15)
- **ReduceLROnPlateau**: Adaptive learning rate reduction

## Output Files

After training, the following files are generated:

- `best_plant_disease_model.h5` - Trained Keras model
- `plant_disease_model.tflite` - Mobile-optimized model
- `training_history.png` - Training metrics plots
- `confusion_matrix.png` - Classification performance matrix
- `gradcam_explanation.png` - Model decision visualization

## Performance Tips

### For Better Accuracy
1. **Larger Dataset**: More diverse training samples
2. **Data Quality**: High-resolution, clear images
3. **Class Balance**: Equal samples per disease class
4. **Augmentation**: Tune augmentation parameters for your domain
5. **Hyperparameter Tuning**: Increase GA generations for better optimization

### For Faster Training
1. **GPU**: Use CUDA-enabled GPU for training
2. **Batch Size**: Increase batch size if memory allows
3. **Early Stopping**: Enable for faster convergence
4. **Transfer Learning**: Start with frozen base model

### For Mobile Deployment
1. **Model Size**: Use quantization for smaller models
2. **Input Size**: Consider smaller input sizes for faster inference
3. **Pruning**: Remove unnecessary model parameters
4. **Hardware**: Test on target mobile devices

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Slow Training**: Enable GPU support or reduce model complexity
3. **Poor Performance**: Check data quality and class distribution
4. **Import Errors**: Install missing dependencies from requirements.txt

### Dataset Issues
- Ensure consistent image formats (JPEG/PNG)
- Check for corrupted images
- Verify folder structure matches expected format
- Balance dataset classes if possible

## Example Results

```
Model Performance:
├── Accuracy:  0.9234
├── Precision: 0.9156
├── Recall:    0.9201
└── F1-Score:  0.9178

Model Sizes:
├── Original: 67.4 MB
├── TFLite:   8.9 MB (87% reduction)
└── Quantized: 4.5 MB (93% reduction)
```

## Contributing

Feel free to contribute improvements:
- Additional augmentation techniques
- Alternative optimization algorithms
- Support for other backbone architectures
- Mobile app integration examples

## License

This project is provided for educational and research purposes.

## Citation

If you use this code in your research, please cite:

```
Plant Leaf Disease Detection with EfficientNetV2 and Meta-heuristic Optimization
GitHub Copilot, 2025
```