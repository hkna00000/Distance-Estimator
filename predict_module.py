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
from ultralytics import YOLO
import cv2
warnings.filterwarnings("ignore")
# Define the Distance Prediction Model (SimpleNN)
class DistanceModel(nn.Module):
    def __init__(self):
        super(DistanceModel, self).__init__()
        # Load pretrained MobileNetV2 and freeze its feature extractor
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        
        # Replace MobileNetV2 classifier with Identity
        num_features = self.backbone.last_channel
        self.backbone.classifier = nn.Identity()
        
        # Define a custom regressor for regression task
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 128),  # Input size matches MobileNetV2's output
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)              # Output a single value for regression
        )
    
    def forward(self, x):
        # Pass input through MobileNetV2 feature extractor
        x = self.backbone.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))  # Ensure output is (batch_size, num_features, 1, 1)
        x = torch.flatten(x, 1)  # Flatten to (batch_size, num_features)
        x = self.regressor(x)    # Pass through regressor
        return x

# Define the Classification Model (SimpleNN)
classify_model = YOLO('model/best.pt')
# Load the Distance Prediction Model
distance_model = DistanceModel()
distance_model.load_state_dict(torch.load('model\mobilevnet2.pth', map_location=torch.device('cpu')))
distance_model.eval()


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

def predict_single_distance(image_path: str, xmin: int, ymin: int, xmax: int, ymax: int):
    """
    Predict the z-location for a given bounding box in an image.
    Args:
        image_path: Path to the input image.
        xmin, ymin, xmax, ymax: Coordinates of the bounding box.
    Returns:
        The predicted z-location for the bounding box.
    """
    try:
        # Load the image to ensure it's valid
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image could not be loaded. Please check the path.")

        # Scale bounding box coordinates from canvas to image resolution
        scale_x = 1242 / 600
        scale_y = 375 / 400

        xmin = int(xmin * scale_x)
        ymin = int(ymin * scale_y)
        xmax = int(xmax * scale_x)
        ymax = int(ymax * scale_y)

         # Crop and preprocess the image
        cropped_image = image[ymin:ymax, xmin:xmax]
        if cropped_image.size == 0:
            raise ValueError("Cropped image has no area. Check bounding box coordinates.")

        # Convert to RGB if the image is grayscale
        if len(cropped_image.shape) == 2:  # Grayscale image
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
        elif cropped_image.shape[2] == 1:  # Single-channel image
            cropped_image = np.repeat(cropped_image, 3, axis=2)

        # Convert to PIL Image and apply VGG16-compatible transforms
        cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        input_tensor = transform(cropped_image).unsqueeze(0)  # Add batch dimension

        # Predict the z-location using the distance model
        with torch.no_grad():
            z_loc_scaled = distance_model(input_tensor).item()

        # Inverse transform to get the original scale
        z_loc = y_scaler.inverse_transform([[z_loc_scaled]])[0, 0]
        return z_loc
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        box1: [xmin, ymin, xmax, ymax]
        box2: [xmin, ymin, xmax, ymax]
    Returns:
        iou: Intersection over Union value
    """
    # Calculate intersection
    ixmin = max(box1[0], box2[0])
    iymin = max(box1[1], box2[1])
    ixmax = min(box1[2], box2[2])
    iymax = min(box1[3], box2[3])

    iw = max(ixmax - ixmin, 0)
    ih = max(iymax - iymin, 0)
    intersection = iw * ih

    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    # Avoid division by zero
    if union == 0:
        return 0

    return intersection / union
def predict_class(image_path: str, xmin: int, ymin: int, xmax: int, ymax: int):
    try:
        # Load the image
        image = cv2.imread(image_path)

        # Scale bounding box coordinates from canvas to image resolution
        scale_x = 1242 / 600
        scale_y = 375 / 400

        xmin = int(xmin * scale_x)
        ymin = int(ymin * scale_y)
        xmax = int(xmax * scale_x)
        ymax = int(ymax * scale_y)
        # Predict using YOLO on the full image
        results = classify_model.predict(source=image_path, save=False, conf=0.1)

        # Check if any boxes are detected
        if not results or not results[0].boxes:
            return "No object detected in the selected region+."

        # Define user-drawn bounding box
        user_box = [xmin, ymin, xmax, ymax]

        # Filter predictions based on IoU
        highest_iou = 0
        predicted_class = None

        for box in results[0].boxes:
            # YOLO bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf.cpu().numpy()[0])  # Confidence score
            cls = int(box.cls.cpu().numpy()[0])     # Class index

            # Calculate IoU with the user-drawn bounding box
            yolo_box = [x1, y1, x2, y2]
            iou = calculate_iou(user_box, yolo_box)

            # Update the prediction if IoU is the highest and above 80%
            if iou > highest_iou and iou >= 0.8:
                highest_iou = iou
                predicted_class = classify_model.names[cls]

        # Return the predicted class if found
        return predicted_class if predicted_class else "No object detected in the selected region."
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Parse arguments and read input
    if len(sys.argv) < 6:
        print("Usage: python combined_module.py <xmin> <ymin> <xmax> <ymax>")
        sys.exit(1)
    image_path = sys.argv[1]  # The first argument is the image path
    xmin, ymin, xmax, ymax = map(int, sys.argv[2:6])

    try:
        # Predict distance
        z_loc = predict_single_distance(image_path, xmin, ymin, xmax, ymax)
        print(f"Predicted z-location: {z_loc}")

        # Predict class
        predicted_class = predict_class(image_path, xmin, ymin, xmax, ymax)
        print(f"Predicted Class: {predicted_class}")
    except Exception as e:
        print(f"Error: {str(e)}")
