import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import RMSprop, AdamW
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from PIL import Image
import os
import cv2
from sklearn.model_selection import train_test_split
import torchvision.models as models
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
y_train = df_train[['zloc']].values

X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
y_test = df_test[['zloc']].values
z_mean, z_std = y_train.mean(), y_train.std()


# Paths
DATA_PATH = "original_data/train_images"
ANNOTATION_FILE = "data/train.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Parameters
IMG_SIZE = (224, 224)  # Resize all cropped regions to 224x224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# Dataset Class
class BoundingBoxDataset(Dataset):
    def __init__(self, annotations, scaler=None, transform=None):
        self.annotations = annotations
        self.transform = transform
        self.scaler = scaler
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_path = os.path.join(DATA_PATH, f"{row['filename'][:-4]}.png")
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Crop the image
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cropped = image[ymin:ymax, xmin:xmax]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        if self.transform:
            cropped = self.transform(cropped)
        
        # Cast label to float32
        label = float(row['zloc'])
        if self.scaler:
            label = self.scaler.transform([[label]])[0][0]
        
        return cropped, torch.tensor(label, dtype=torch.float32)
def validate_images():
    annotations = pd.read_csv(ANNOTATION_FILE)
    missing_files = []
    for filename in annotations['filename'].unique():
        image_path = os.path.join(DATA_PATH, f"{filename[:-4]}.png")
        if not os.path.exists(image_path):
            missing_files.append(image_path)
    
    if missing_files:
        print(f"Missing images: {len(missing_files)}")
        for path in missing_files[:10]:  # Show first 10 missing files
            print(path)
    else:
        print("All image files are present.")

validate_images()
# Load and preprocess data
def load_data():
    df = pd.read_csv(ANNOTATION_FILE)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Standardize labels
    scaler = StandardScaler()
    train_df['zloc'] = scaler.fit_transform(train_df[['zloc']])
    test_df['zloc'] = scaler.transform(test_df[['zloc']])
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = BoundingBoxDataset(train_df, transform=transform)
    test_dataset = BoundingBoxDataset(test_df, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, scaler

# Model Definition
class ZLocModel(nn.Module):
    def __init__(self):
        super(ZLocModel, self).__init__()
        self.backbone = models.vgg16(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False  # Freeze backbone
        self.backbone.classifier = nn.Sequential(
            nn.Linear(25088, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Training Function
def train_model(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            # Move data to the appropriate device and type
            images, labels = images.to(DEVICE, dtype=torch.float32), labels.to(DEVICE, dtype=torch.float32)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# Evaluation Function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss:.4f}")

# Main Function
def main():
    # Load data
    train_loader, test_loader, scaler = load_data()
    
    # Initialize model, loss, and optimizer
    model = ZLocModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    train_model(model, train_loader, optimizer, criterion, EPOCHS)
    
    # Evaluate the model
    evaluate_model(model, test_loader, criterion)
    
    # Save the model
    torch.save(model.state_dict(), "model/zloc_predictor.pth")
    print("Model saved to model/zloc_predictor.pth")
    
if __name__ == "__main__":
    main()