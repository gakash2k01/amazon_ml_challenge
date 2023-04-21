import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.score import score_function

# Define a polynomial regressor model
class MyModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=3, output_size=1):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx].astype(np.float32))
        x = x.unsqueeze(0)
        y = torch.tensor(self.label.iloc[idx].astype(np.float32))

        return x, y
    
if __name__ == "__main__":
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on:", device)



    # Define dataset and data loader
    print("Reading data.")
    train_data = pd.read_csv('dataset/train.csv')
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    x_train = train_data['PRODUCT_TYPE_ID']
    y_train = train_data['PRODUCT_LENGTH']
    x_val = val_data['PRODUCT_TYPE_ID']
    y_val = val_data['PRODUCT_LENGTH'].clip(upper=5000)
    train_dataset = CustomDataset(data=x_train, label=y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_dataset = CustomDataset(data=x_val, label=y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    # Initialize model, optimizer, and loss function
    print("Creating model.")
    model = MyModel()

    # Move the model to the device
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=8e-6)
    criterion = nn.MSELoss()

    # Train model using batch-wise training
    n_epochs = 100

    for epoch in range(n_epochs):
        train_loss = 0.0
        
        for X_batch, y_batch in tqdm(train_dataloader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_dataset)
        
        # Evaluate model on test set
        test_loss = 0.0
        val_score = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_dataloader):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred.squeeze(), y_batch)
                test_loss += loss.item() * X_batch.size(0)
                val_score += (score_function(y_batch.cpu().numpy(), y_pred.cpu().numpy()) * len(y_batch))
            test_loss /= len(val_dataset)
            val_score /= len(val_dataset)

        print("Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Val Score: {:.4f}".format(epoch+1, n_epochs, train_loss, test_loss, val_score))
