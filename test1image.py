import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Define your SimpleNN model class (if it’s not imported)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
model = SimpleNN()
model.load_state_dict(torch.load('model/simple_nn_model.pth'))
model.eval()

# Load scalers
X_scaler = joblib.load('X_scaler.pkl')
y_scaler = joblib.load('y_scaler.pkl')

# Single sample input
sample = pd.DataFrame([[633, 155, 704, 211]], columns=['xmin', 'ymin', 'xmax', 'ymax'])

# Preprocess the sample
sample_scaled = X_scaler.transform(sample)
sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

# Predict with the model
with torch.no_grad():
    z_loc_scaled = model(sample_tensor).item()

# Inverse transform to get original scale
z_loc = y_scaler.inverse_transform([[z_loc_scaled]])[0, 0]

print(f"Predicted z location: {z_loc}")
