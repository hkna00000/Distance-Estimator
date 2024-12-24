import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import warnings
import sys
import io
warnings.filterwarnings("ignore")
# Load ResNet18 as a feature extractor
resnet = models.resnet18(pretrained=True)
# Modify the last fully connected layer to output 2048 features
resnet.fc = nn.Linear(resnet.fc.in_features, 2048)
resnet.eval()

# Load the SimpleNN model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
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

# Ensure you define your SimpleNN structure
model = SimpleNN(input_dim=2048, num_classes=8) 
model.load_state_dict(torch.load("model/simple_nn_classi.pth"))
model.eval()

# Load the label encoder (use the one from training)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("scaler/class_names.npy")  # Replace with actual path if saved during training

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match input size for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_class(image_data: bytes, xmin: int, ymin: int, xmax: int, ymax: int):
    # Convert image bytes to PIL image
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Crop the image to the bounding box
    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    cropped_image = transform(cropped_image).unsqueeze(0)  # Add batch dimension

    # Extract features using ResNet (without the classification layer)
    with torch.no_grad():
        feature = resnet(cropped_image).squeeze()  # Remove extra dimensions
        feature = feature.view(-1)  # Flatten the feature map to 1D (2048)

    # Convert to tensor for SimpleNN
    feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Make prediction with the SimpleNN model
    with torch.no_grad():
        outputs = model(feature_tensor)
        _, predicted_class_idx = torch.max(outputs, 1)

    # Map the predicted class index to class label (ensure you load your label encoder properly)
    predicted_class_label = label_encoder.inverse_transform([predicted_class_idx.item()])
    return predicted_class_label[0]

if __name__ == "__main__":
    # Read image data from stdin
    input_image_data = sys.stdin.read()  # Read everything from stdin

    # Parse the bounding box arguments
    if len(sys.argv) < 6:
        print("Usage: python classi_module.py <xmin> <ymin> <xmax> <ymax>")
        sys.exit(1)

    xmin, ymin, xmax, ymax = map(int, sys.argv[1:5])

    # Predict the class based on the image data and bounding box
    result = predict_class(input_image_data, xmin, ymin, xmax, ymax)
    print(f"Predicted Class: {result}")