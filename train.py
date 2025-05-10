import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from model import BloodTransfusionModel
import matplotlib.pyplot as plt
import os
from PIL import Image

def load_dataset(images_dir, masks_dir, is_segmentation=True):
    # Load images
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    
    images = []
    masks = []
    
    for image_file in image_files:
        # Load image
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image_tensor = torch.FloatTensor(np.array(image)) / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension [1, H, W]
        images.append(image_tensor)
        
        # Load corresponding mask
        mask_path = os.path.join(masks_dir, image_file)
        mask = Image.open(mask_path).convert('L')
        
        if is_segmentation:
            # For segmentation: keep mask as image
            mask_tensor = torch.FloatTensor(np.array(mask)) / 255.0
            mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension [1, H, W]
            masks.append(mask_tensor)
        else:
            # For classification: convert mask to class index
            # Assuming binary classification based on presence of white pixels
            mask_np = np.array(mask)
            class_index = 1 if np.sum(mask_np) > 0 else 0
            masks.append(class_index)
    
    # Stack images
    X = torch.stack(images)
    
    if is_segmentation:
        y = torch.stack(masks)
    else:
        y = torch.LongTensor(masks)
    
    return X, y

def train_model(train_loader, val_loader, batch_size=32, epochs=100, learning_rate=0.001):
    # Initialize model
    model = BloodTransfusionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            print('-' * 50)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    return model, train_losses, val_losses, train_accs, val_accs

def test_model(model, test_loader):
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()
    
    # Lists to store predictions and ground truth for visualization
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
            
            # Store predictions and targets for visualization
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Calculate final metrics
    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total
    
    print('\nTest Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Visualize some predictions
    plt.figure(figsize=(15, 5))
    for i in range(min(5, len(all_predictions))):
        plt.subplot(1, 5, i+1)
        plt.imshow(all_predictions[i], cmap='gray')
        plt.title(f'Predicted\nAccuracy: {test_acc:.2f}%')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    plt.close()
    
    return test_loss, test_acc

if __name__ == "__main__":
    train_images_dir = os.path.join('datasets', 'train', 'images')
    val_images_dir = os.path.join('datasets', 'val', 'images')
    test_images_dir = os.path.join('datasets', 'test', 'images')

    train_masks_dir = os.path.join('datasets', 'train', 'masks')
    val_masks_dir = os.path.join('datasets', 'val', 'masks')
    test_masks_dir = os.path.join('datasets', 'test', 'masks')

    # Load datasets
    X_train, y_train = load_dataset(train_images_dir, train_masks_dir)
    X_val, y_val = load_dataset(val_images_dir, val_masks_dir)
    X_test, y_test = load_dataset(test_images_dir, test_masks_dir)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Train the model
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        train_loader, val_loader,
        batch_size=64,
        epochs=100,
        learning_rate=0.001
    )
    
    # Save the model
    torch.save(model.state_dict(), 'blood_transfusion_model.pth')
    
    # Test the model
    test_loss, test_acc = test_model(model, test_loader) 

    #save model after test
    torch.save(model.state_dict(), "final_bloodTransfusionModel.pth")
