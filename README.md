# Blood Transfusion AI Model

This repository contains a PyTorch implementation of a neural network model for blood transfusion prediction. The model uses a multi-layer architecture with batch normalization and dropout for improved performance.

## Features

- Multi-layer neural network architecture
- Batch normalization for faster training
- Dropout for regularization
- Learning rate scheduling
- Gradient clipping
- Training history visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/blood-transfusion-ai.git
cd blood-transfusion-ai
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your data in the correct format (X: features, y: labels)

2. Train the model:
```python
from train import train_model
from sklearn.preprocessing import StandardScaler

# Scale your features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model, train_losses, val_losses, train_accs, val_accs = train_model(
    X_scaled, y,
    batch_size=64,
    epochs=100,
    learning_rate=0.001
)
```

3. Save the trained model:
```python
torch.save(model.state_dict(), 'blood_transfusion_model.pth')
```

## Model Architecture

The model consists of:
- Input layer with batch normalization
- Three hidden layers with batch normalization and dropout
- Output layer for classification

## Training Process

The training process includes:
- Learning rate scheduling with ReduceLROnPlateau
- Gradient clipping to prevent exploding gradients
- Validation monitoring
- Training history visualization

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. "# VeinDetectionModel" 
