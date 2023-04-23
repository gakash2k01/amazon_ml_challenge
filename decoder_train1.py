import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
import pathlib, os
import warnings
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from lion_pytorch import Lion
from utils.score import score_function

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_everything(seed):
    '''
    Seeds everything so as to allow for reproducibility
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff1 = nn.Linear(13421,128)
        self.ff2 = nn.Linear(768+128, 120)
        self.ff3 = nn.Linear(120, 24)
        self.ff4 = nn.Linear(24, 1)
        self.relu = nn.ReLU()
        

    def forward(self, X, X1):
        X1 = self.ff1(X1)
        X1 = self.relu(X1)
        X2 = torch.cat((X, X1), dim=1)
        X2 = X2.to(self.ff2.weight.dtype)
        output = self.ff2(X2)
        output = self.relu(output)
        output = self.ff3(output)
        output = self.relu(output)
        output = self.ff4(output)
        return output

class CustomDataset(Dataset):
    def __init__(self, data, data1, label):
        self.data = data
        self.data1 = data1
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        x_i = self.data.iloc[ind]
        x1_i = self.data1.iloc[ind]
        y_i = self.label.iloc[ind]

        return torch.tensor(x_i), torch.tensor(x1_i), torch.tensor(y_i)

    
def train(model, iterator, optimizer):
    '''
    function: Training the model
    Input: model, iterator, optimizer
    Returns: epoch_loss
    '''
    epoch_loss = 0
    criterion = nn.MSELoss()
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    
    total = 0
    for batch_data in tqdm(iterator):
        x_i, x1_i, y_i = batch_data
        x_i = x_i.to(device).float()
        x1_i = x1_i.to(device).float()
        y_i = y_i.to(device).float()
        optimizer.zero_grad()
        # Obtaining the logits to obtain the loss
        predictions = model(x_i, x1_i)
        loss = criterion(predictions, y_i)
        loss.backward()
        optimizer.step()
        total+=len(y_i)
        epoch_loss += loss.item()
    
    return epoch_loss / total

def evaluate(model, iterator):
    '''
    function: Evaluating the model
    Input: model, iterator, pad_id
    Returns: epoch_loss, epoch_acc
    '''
    model.eval()
    final_pred = []
    # Predicted value
    comp_pred = []
    with torch.no_grad():
        for batch_data in tqdm(iterator):
            x_i, x1_i, y_i = batch_data
            x_i = x_i.to(device).float()
            x1_i = x1_i.to(device).float()
            y_i = y_i.to(device).float()
            model.to(device)
            predictions = model(x_i, x1_i)
            predictions = predictions.squeeze().cpu().numpy()
            final_pred += list(predictions)
            comp_pred += list(y_i.cpu().numpy())
    final_pred = torch.tensor(final_pred)
    comp_pred = torch.tensor(comp_pred)
    score = score_function(final_pred, comp_pred)
    error = ((abs(np.array(final_pred)-np.array(comp_pred))).sum())/len(final_pred)
    return score, error

def run(model, root_dir):
    '''
    Function: Similar to the main function
    '''
    torch.cuda.empty_cache()
    seed_everything(SEED)
    print("Setting up data.")
    data = pd.read_csv(f'{root_dir}/embeddings1.csv')
    data1 = pd.read_csv(f'{root_dir}/mini_train1.csv')
    data_train, data_val, data1_train, data1_val = train_test_split(data, data1, test_size=0.2, random_state=42)
    y_train = data1_train['PRODUCT_LENGTH']
    y_val = data1_val['PRODUCT_LENGTH']
    
    # create a sample DataFrame with a categorical column containing values from 0 to 13420
    df = pd.DataFrame({
        'PRODUCT_TYPE_ID': range(13421),
        'value': [i**2 for i in range(13421)]
    })
    encoder = OneHotEncoder(sparse=False)
    _ = encoder.fit_transform(df[['PRODUCT_TYPE_ID']])
    onehot_array1 = encoder.transform(pd.DataFrame(data1_train['PRODUCT_TYPE_ID']))
    train_onehot_df = pd.DataFrame(onehot_array1, columns=encoder.get_feature_names_out(['PRODUCT_TYPE_ID']))
    onehot_array2 = encoder.transform(pd.DataFrame(data1_val['PRODUCT_TYPE_ID']))
    val_onehot_df = pd.DataFrame(onehot_array2, columns=encoder.get_feature_names_out(['PRODUCT_TYPE_ID']))
    # Dataloader
    train_dataset = CustomDataset(data = data_train, data1 = train_onehot_df, label = y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = num_workers, drop_last = False)
    val_dataset = CustomDataset(data = data_val, data1 = val_onehot_df, label = y_val)
    valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = num_workers, drop_last = False)
    optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = ExponentialLR(optimizer, gamma=0.8)
    model = model.to(device)

    N_EPOCHS = num_epoch
    best_acc = 0.0
    for epoch in range(N_EPOCHS): 
        #training part
        train_loss = train(model, train_loader, optimizer)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Validation
        valid_loss, val_mae = evaluate(model, valid_loader)
        print(f'Val. mae: {val_mae:.2f}\t Val. score: {valid_loss:.2f}%')
        if(valid_loss>best_acc):
            best_acc = valid_loss
            torch.save(model.state_dict(), f'{base_path}/weights/best_model4.pth')
            print("Model saved.")
        if wandb_login:
            wandb.log({"Training loss": train_loss, "Validation MAE": val_mae, "Validation Score": valid_loss, "lr": scheduler.get_last_lr()[0]}, step=epoch)
        scheduler.step()


if __name__ == "__main__":
    # Helps make all paths relative
    base_path = pathlib.Path().absolute()

    # Input of the required hyperparameters
    BATCH_SIZE = 120 *2
    learning_rate = 5e-3
    device = 'cuda:2'
    wandb_login = True

    SEED = 42
    num_epoch = 100
    num_workers = 4
    # Path to the dataset
    root_dir = f"{base_path}/dataset"
    if not os.path.exists(root_dir):
        print("Dataset missing.")
    
    #Loading the pretrained model and tokenizer
    
    model = Decoder()
    if wandb_login:
        # For logging
        wandb.login()
        wandb.init(project="amazon_ml_challenge", entity="gakash2001")
        wandb.config = {
            "learning_rate": learning_rate,
            "epochs": num_epoch,
            "batch_size": BATCH_SIZE
        }
    run(model, root_dir)
