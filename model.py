import torch
import torch.nn as nn
import torch.nn.functional as F

class BloodTransfusionModel(nn.Module):
    def __init__(self, num_channels=1, num_classes=2):
        super(BloodTransfusionModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate output size after convolutions (depends on input image size)
        # For example, if input is 224x224:
        # After 3 pooling layers: 224 -> 112 -> 56 -> 28
        # So feature map size would be 28x28 with 128 channels
        self.fc_input_size = 128 * 28 * 28  # Adjust based on your input image size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Input shape: [batch_size, channels, height, width]
        
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
