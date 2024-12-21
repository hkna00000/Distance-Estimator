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

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
y_train = df_train[['zloc']].values

X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
y_test = df_test[['zloc']].values
z_mean, z_std = y_train.mean(), y_train.std()
outliers = (y_train < (z_mean - 3 * z_std)) | (y_train > (z_mean + 3 * z_std))

# Flatten the mask if y_train is a 2D array
outliers = outliers.flatten()
# Filter out rows in X_train and y_train based on the outliers mask
X_train_filtered = X_train[~outliers]
y_train_filtered = y_train[~outliers]
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
joblib.dump(X_scaler, 'X_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')
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
    
def load_model(model_path):
    model = SimpleNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

class CustomDataset(Dataset):
    def __init__(self, bboxes, zlocs):
        self.bboxes = bboxes  # Bounding boxes (X_train or X_test)
        self.zlocs = zlocs    # Z-locations (y_train or y_test)

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        bbox = torch.tensor(self.bboxes[idx], dtype=torch.float32)  # Convert bounding box to tensor
        zloc = torch.tensor(self.zlocs[idx], dtype=torch.float32)    # Convert z-location to tensor
        return bbox, zloc


from torchvision import transforms

# Define transformations (e.g., resizing, normalization)
transform = transforms.Compose([
    transforms.Resize((375, 1242)),  # Resize to match your image dimensions
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB channels
])

# Assuming you have a list of image file paths
image_paths_train = "original_data/train_images"  # Load your training image paths
image_paths_test = "original_data/test_images"  # Load your test image paths

# Create datasets for training and testing
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

# Create DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


model = SimpleNN()  # Initialize the neural network model
criterion = nn.MSELoss()  # Loss function for regression
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-5) #optimizer

# Training Loop
num_epochs = 100  # Increase epochs as needed
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for bboxes, zlocs in train_loader:
        optimizer.zero_grad()  # Clear gradients
        outputs = model(bboxes)  # Forward pass
        loss = criterion(outputs, zlocs)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()  # Set the model to evaluation mode
total_loss = 0.0
with torch.no_grad():
    for bboxes, zlocs in test_loader:
        outputs = model(bboxes)
        loss = criterion(outputs, zlocs)
        total_loss += loss.item()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    y_pred = model(X_test_tensor).numpy()

# Inverse transform the scaled predictions and ground truth
y_pred = y_scaler.inverse_transform(y_pred)
y_test = y_scaler.inverse_transform(y_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")
print(f"Test Loss: {total_loss/len(test_loader):.4f}")


print(f"Now making predictions on the test set...")
df_test = pd.read_csv('data/test.csv')
X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
y_test = df_test[['zloc']].values  # True values for comparison
X_scaler = joblib.load('X_scaler.pkl')
y_scaler = joblib.load('y_scaler.pkl')
X_test = X_scaler.fit_transform(X_test)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy().flatten()
y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
df_test['zloc_pred'] = y_pred
df_test.to_csv('data/predictions_simple_nn.csv', index=False)
# Save the PyTorch model
torch.save(model.state_dict(), 'model/simple_nn_model.pth')
print("SimpleNN model saved to 'model/simple_nn_model.pth'")
print("Predictions saved to 'data/predictions_simple_nn.csv'")
###########################################################################################

print("Loading the Random Forest model")
from sklearn.ensemble import RandomForestRegressor

# Assuming you have your data loaded and split
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
y_train = df_train['zloc'].values
X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
y_test = df_test['zloc'].values

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

print(f"Now making predictions on the test set...")
df_test = pd.read_csv('data/test.csv')
X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
y_test = df_test[['zloc']].values  # True values for comparison
y_pred = rf_model.predict(X_test)
# Check if predictions are constant
print("Predictions before inverse scaling:", y_pred[:10])
if not np.all(y_pred == y_pred[0]):
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
else:
    print("Warning: Predictions are constant before inverse scaling.")
    y_pred = y_pred  # Skip scaling if already constant
# Verify predictions in the final output
print("Final predictions:", y_pred[:10])
df_test['zloc_pred'] = y_pred
# Save to CSV
df_test.to_csv('data/predictions_random_forest.csv', index=False)
print("Predictions saved to 'data/predictions_random_forest.csv'")