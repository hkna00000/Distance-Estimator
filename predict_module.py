import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import warnings
import sys
import io
warnings.filterwarnings("ignore")
# Define your SimpleNN model class 
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Input layer (4 features: xmin, ymin, xmax, ymax)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)    # Output layer (predicting z-location)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # First layer with ReLU activation
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # Second layer with ReLU activation
        x = self.fc3(x)          # Output layer
        return x

# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load('model/simple_nn_model.pth',  weights_only=True))
model.eval()
xmin = int(sys.argv[1])
ymin = int(sys.argv[2])
xmax = int(sys.argv[3])
ymax = int(sys.argv[4])
# Load scalers
X_scaler = joblib.load('scaler/X_scaler.pkl')
y_scaler = joblib.load('scaler/y_scaler.pkl')

# Single sample input
sample = pd.DataFrame([[xmin,ymin,xmax,ymax]], columns=['xmin', 'ymin', 'xmax', 'ymax'])

# Preprocess the sample
sample_scaled = X_scaler.transform(sample)
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

# Predict with the model
with torch.no_grad():
    z_loc_scaled = model(sample_tensor).item()

# Inverse transform to get original scale
z_loc = y_scaler.inverse_transform([[z_loc_scaled]])[0, 0]

print(f"Predicted z location: {z_loc}")

