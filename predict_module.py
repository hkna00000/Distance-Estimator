import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
import warnings
import sys
import io
warnings.filterwarnings("ignore")
# Define the Distance Prediction Model (SimpleNN)
class DistanceModel(nn.Module):
    def __init__(self):
        super(DistanceModel, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Input layer (4 features: xmin, ymin, xmax, ymax)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)  # Output layer (predicting z-location)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # First layer with ReLU activation
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))  # Second layer with ReLU activation
        x = self.fc3(x)  # Output layer
        return x

# Define the Classification Model
class ClassificationModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Load the Distance Prediction Model
distance_model = DistanceModel()
distance_model.load_state_dict(torch.load('model/simple_nn_model.pth', map_location=torch.device('cpu')))
distance_model.eval()

# Load the Classification Model
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 2048)
resnet.eval()
classification_model = ClassificationModel(input_dim=2048, num_classes=8)
classification_model.load_state_dict(torch.load("model/simple_nn_classi.pth", map_location=torch.device('cpu')))
classification_model.eval()

# Load the scalers and label encoder
X_scaler = joblib.load('scaler/X_scaler.pkl')
y_scaler = joblib.load('scaler/y_scaler.pkl')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("scaler/class_names.npy") 
# Define the image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def predict_distance(xmin, ymin, xmax, ymax):
    # Prepare the input for the distance model
    sample = pd.DataFrame([[xmin, ymin, xmax, ymax]], columns=['xmin', 'ymin', 'xmax', 'ymax'])
    sample_scaled = X_scaler.transform(sample)
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

    # Predict with the distance model
    with torch.no_grad():
        z_loc_scaled = distance_model(sample_tensor).item()

    # Inverse transform to get original scale
    z_loc = y_scaler.inverse_transform([[z_loc_scaled]])[0, 0]
    return z_loc

def predict_class(image_path: bytes, xmin, ymin, xmax, ymax):
    try:
        # Open the image directly from the file path
        image = Image.open(image_path).convert("RGB")

        # Crop and transform the image
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        cropped_image = transform(cropped_image).unsqueeze(0)

        with torch.no_grad():
            features = resnet(cropped_image).squeeze().view(-1)

        feature_tensor = features.unsqueeze(0)
        with torch.no_grad():
            outputs = classification_model(feature_tensor)
            _, predicted_class_idx = torch.max(outputs, 1)

        predicted_class_label = label_encoder.inverse_transform([predicted_class_idx.item()])
        return predicted_class_label[0]
    except Exception as e:
        print(f"Error: {e}")
        return "Error in processing image"

if __name__ == "__main__":
    # Parse arguments and read input
    if len(sys.argv) < 6:
        print("Usage: python combined_module.py <xmin> <ymin> <xmax> <ymax>")
        sys.exit(1)
    image_path = sys.argv[1]  # The first argument is the image path
    xmin, ymin, xmax, ymax = map(int, sys.argv[2:6])

    try:
        # Predict distance
        z_loc = predict_distance(xmin, ymin, xmax, ymax)
        print(f"Predicted z-location: {z_loc}")

        # Predict class
        predicted_class = predict_class(image_path, xmin, ymin, xmax, ymax)
        print(f"Predicted Class: {predicted_class}")
    except Exception as e:
        print(f"Error: {str(e)}")